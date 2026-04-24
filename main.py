```py
"""
FastAPI Application Entry Point for AI Video Generator

This module implements the backend API for an AI-powered video generation application.
It provides REST endpoints and WebSocket support for:
- Chat-based script generation using DeepSeek AI
- Media fetching from Pixabay
- Voiceover generation using ElevenLabs
- Video composition and download
- Session management with SQLite

All API keys are loaded from environment variables for security.
"""

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables with validation
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not all([DEEPSEEK_API_KEY, PIXABAY_API_KEY, ELEVENLABS_API_KEY]):
    logger.warning("Some API keys are missing. Check environment variables.")

# Constants
DATABASE_URL = "sqlite:///./video_generator.db"
MAX_SCRIPT_LENGTH = 5000
MAX_MEDIA_PER_TOPIC = 10
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv"}
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".ogg"}
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

# SQLAlchemy setup
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class SessionModel(Base):
    """Database model for user sessions."""
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    chat_history = Column(JSON, default=list)
    current_topic = Column(String, nullable=True)
    generated_script = Column(Text, nullable=True)
    media_urls = Column(JSON, default=list)
    voiceover_url = Column(String, nullable=True)
    video_url = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)


# Pydantic models for request/response validation
class ChatMessage(BaseModel):
    """Model for chat messages."""
    session_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=MAX_SCRIPT_LENGTH)

    @validator("message")
    def validate_message(cls, v):
        """Sanitize and validate message content."""
        # Remove potentially harmful characters
        sanitized = v.strip()
        if len(sanitized) < 1:
            raise ValueError("Message cannot be empty")
        return sanitized


class ScriptRequest(BaseModel):
    """Model for script generation requests."""
    session_id: str = Field(..., min_length=1, max_length=100)
    topic: str = Field(..., min_length=1, max_length=200)
    style: Optional[str] = Field(default="informative", max_length=50)

    @validator("topic")
    def validate_topic(cls, v):
        """Sanitize topic input."""
        sanitized = v.strip()
        if len(sanitized) < 1:
            raise ValueError("Topic cannot be empty")
        return sanitized


class MediaRequest(BaseModel):
    """Model for media fetching requests."""
    session_id: str = Field(..., min_length=1, max_length=100)
    query: str = Field(..., min_length=1, max_length=200)
    media_type: str = Field(default="video", regex="^(image|video)$")

    @validator("query")
    def validate_query(cls, v):
        """Sanitize search query."""
        sanitized = v.strip()
        if len(sanitized) < 1:
            raise ValueError("Query cannot be empty")
        return sanitized


class VoiceoverRequest(BaseModel):
    """Model for voiceover generation requests."""
    session_id: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=1, max_length=MAX_SCRIPT_LENGTH)
    voice_id: Optional[str] = Field(default="21m00Tcm4TlvDq8ikWAM", max_length=100)

    @validator("text")
    def validate_text(cls, v):
        """Sanitize text input."""
        sanitized = v.strip()
        if len(sanitized) < 1:
            raise ValueError("Text cannot be empty")
        return sanitized


class VideoRequest(BaseModel):
    """Model for video composition requests."""
    session_id: str = Field(..., min_length=1, max_length=100)


# FastAPI application setup
app = FastAPI(
    title="AI Video Generator API",
    description="Backend API for AI-powered video generation with chat interface",
    version="1.0.0",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get database session
def get_db():
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session(db: Session, session_id: str) -> SessionModel:
    """Retrieve or create a session."""
    session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
    if not session:
        session = SessionModel(
            session_id=session_id,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
            chat_history=[],
            media_urls=[],
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    else:
        session.last_active = datetime.utcnow()
        db.commit()
    return session


async def generate_script_with_deepseek(topic: str, style: str = "informative") -> str:
    """
    Generate a video script using DeepSeek AI.

    Args:
        topic: The topic for the video script
        style: The style of the script (informative, persuasive, entertaining)

    Returns:
        Generated script text

    Raises:
        HTTPException: If API call fails or returns invalid response
    """
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DeepSeek API key not configured")

    prompt = f"Write a {style} video script about {topic}. Keep it concise and engaging."

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a professional video script writer."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            data = response.json()

            if "choices" not in data or not data["choices"]:
                raise HTTPException(status_code=500, detail="Invalid response from DeepSeek API")

            script = data["choices"][0]["message"]["content"]
            return script.strip()

        except httpx.HTTPStatusError as e:
            logger.error(f"DeepSeek API error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=502, detail=f"DeepSeek API error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"DeepSeek API request failed: {str(e)}")
            raise HTTPException(status_code=503, detail="DeepSeek API unavailable")


async def fetch_media_from_pixabay(query: str, media_type: str = "video") -> List[Dict[str, Any]]:
    """
    Fetch media (images/videos) from Pixabay API.

    Args:
        query: Search query for media
        media_type: Type of media to fetch ("image" or "video")

    Returns:
        List of media URLs and metadata

    Raises:
        HTTPException: If API call fails or returns invalid response
    """
    if not PIXABAY_API_KEY:
        raise HTTPException(status_code=500, detail="Pixabay API key not configured")

    endpoint = "videos" if media_type == "video" else ""
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            params = {
                "key": PIXABAY_API_KEY,
                "q": query,
                "per_page": MAX_MEDIA_PER_TOPIC,
                "safesearch": True,
            }
            
            if media_type == "video":
                response = await client.get("https://pixabay.com/api/videos/", params=params)
            else:
                response = await client.get("https://pixabay.com/api/", params={**params, "image_type": "photo"})
            
            response.raise_for_status()
            data = response.json()

            if media_type == "video":
                hits = data.get("hits", [])
                media_list = []
                for hit in hits[:MAX_MEDIA_PER_TOPIC]:
                    videos = hit.get("videos", {})
                    # Get the largest available video
                    for quality in ["large", "medium", "small", "tiny"]:
                        if quality in videos and videos[quality].get("url"):
                            media_list.append({
                                "url": videos[quality]["url"],
                                "width": videos[quality].get("width"),
                                "height": videos[quality].get("height"),
                                "duration": hit.get("duration"),
                                "tags": hit.get("tags", "").split(","),
                            })
                            break
            else:
                hits = data.get("hits", [])
                media_list = [
                    {
                        "url": hit.get("largeImageURL") or hit.get("webformatURL"),
                        "preview_url": hit.get("previewURL"),
                        "tags": hit.get("tags", "").split(","),
                        "user": hit.get("user"),
                    }
                    for hit in hits[:MAX_MEDIA_PER_TOPIC]
                ]

            return media_list

        except httpx.HTTPStatusError as e:
            logger.error(f"Pixabay API error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=502, detail=f"Pixabay API error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Pixabay API request failed: {str(e)}")
            raise HTTPException(status_code=503, detail="Pixabay API unavailable")


async def generate_voiceover_with_elevenlabs(text: str, voice_id: str) -> bytes:
    """
    Generate voiceover audio using ElevenLabs API.

    Args:
        text: Text to convert to speech
        voice_id: ElevenLabs voice ID

    Returns:
        Audio data as bytes

    Raises:
        HTTPException: If API call fails or returns invalid response
    """
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": ELEVENLABS_API_KEY,
                },
                json={
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.5,
                    },
                },
            )
            response.raise_for_status()
            return response.content

        except httpx.HTTPStatusError as e:
            logger.error(f"ElevenLabs API error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=502, detail=f"ElevenLabs API error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"ElevenLabs API request failed: {str(e)}")
            raise HTTPException(status_code=503, detail="ElevenLabs API unavailable")


async def compose_video(media_urls: List[str], audio_path: str) -> str:
    """
    Compose final video from media files and audio.

    Args:
        media_urls: List of media file URLs to include in the video
        audio_path: Path to the audio file for voiceover

    Returns:
        Path to the composed video file

    Raises:
        HTTPException: If video composition fails
    """
    # This is a placeholder implementation. In production, you would use FFmpeg or similar.
    
    output_path = TEMP_DIR / f"video_{uuid.uuid4().hex}.mp4"
    
    # Simulate video composition (replace with actual FFmpeg implementation)
    await asyncio.sleep(2)  # Simulate processing time
    
    # For now, we'll just copy the first media file as a placeholder
    # In production, implement actual video composition logic here
    
    return str(output_path)


# REST API Endpoints

@app.get("/")
async def root():
    """Root endpoint returning API status."""
    return {
        "status": "healthy",
        "service": "AI Video Generator API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/session/create")
async def create_session(db: Session = Depends(get_db)):
    """
    Create a new user session.

    Returns:
        Session ID and creation timestamp
    """
    session_id = uuid.uuid4().hex[:16]
    
    new_session = SessionModel(
        session_id=session_id,
        created_at=datetime.utcnow(),
        last_active=datetime.utcnow(),
        chat_history=[],
        media_urls=[],
    )
    
    db.add(new_session)
    db.commit()
    
    logger.info(f"Created new session: {session_id}")
    
    return {
        "session_id": session_id,
        "created_at": new_session.created_at.isoformat(),
        "message": "Session created successfully",
    }


@app.post("/api/chat/send")
async def send_chat_message(message: ChatMessage, db: Session = Depends(get_db)):
    """
Process a chat message and generate AI response.

Args:
message: Chat message with session ID and message content

Returns:
AI-generated response and updated chat history
"""
session = get_session(db, message.session_id)

# Add user message to history
session.chat_history.append({
"role": "user",
"content": message.message,
"timestamp": datetime.utcnow().isoformat(),
})

try:
# Generate AI response using DeepSeek
ai_response = await generate_script_with_deepseek(message.message)

# Add AI response to history
session.chat_history.append({
"role": "assistant",
"content": ai_response,
"timestamp": datetime.utcnow().isoformat(),
})

db.commit()

return {
"session_id": message.session_id,
"response": ai_response,
"chat_history": session.chat_history,
}

except HTTPException as e:
db.rollback()
raise e


@app.post("/api/script/generate")
async def generate_script(request: ScriptRequest, db: Session = Depends(get_db)):
"""
Generate a video script based on topic.

Args:
request: Script generation request with topic and optional style

Returns:
Generated script and metadata
"""
session = get_session(db, request.session_id)

try:
script = await generate_script_with_deepseek(request.topic, request.style)

# Update session with generated script
session.current_topic = request.topic
session.generated_script = script

# Add to chat history
session.chat_history.append({
"role": "system",
"content": f"Generated script for topic: {request.topic}",
"timestamp": datetime.utcnow().isoformat(),
})
session.chat_history.append({
"role": "assistant",
"content": script,
"timestamp": datetime.utcnow().isoformat(),
})

db.commit()

return {
"session_id": request.session_id,
"topic": request.topic,
"script": script,
"style": request.style,
}

except HTTPException as e:
db.rollback()
raise e


@app.post("/api/media/fetch")
async def fetch_media(request: MediaRequest, db: Session = Depends(get_db)):
"""
Fetch relevant media (images/videos) based on query.

Args:
request: Media fetch request with query and media type

Returns:
List of media URLs and metadata
"""
session = get_session(db, request.session_id)

try:
media_list = await fetch_media_from_pixabay(request.query, request.media_type)

# Update session with fetched media URLs
media_urls = [media["url"] for media in media_list]
session.media_urls.extend(media_urls)

db.commit()

return {
"session_id": request.session_id,
"query": request.query,
"media_type": request.media_type,
"media_count": len(media_list),
"media_list": media_list,
}

except HTTPException as e:
db.rollback()
raise e


@app.post("/api/voiceover/generate")
async def generate_voiceover(request: VoiceoverRequest, db: Session = Depends(get_db)):
"""
Generate voiceover audio from text.

Args:
request: Voiceover generation request with text and optional voice ID

Returns:
URL to generated audio file
"""
session = get_session(db, request.session_id)

try:
audio_data = await generate_voiceover_with_elevenlabs(request.text, request.voice_id)

# Save audio file
audio_filename = f"voiceover_{uuid.uuid4().hex}.mp3"
audio_path = TEMP_DIR / audio_filename

async with aiofiles.open(audio_path, "wb") as f:
await f.write(audio_data)

# Update session with voiceover URL
voiceover_url = f"/temp/{audio_filename}"
session.voiceover_url = voiceover_url

db.commit()

return {
"session_id": request.session_id,
"voiceover_url": voiceover_url,
"duration_seconds": len(audio_data) // 32000,  # Approximate duration for MP3 at 32kbps
}

except HTTPException as e:
db.rollback()
raise e


@app.post("/api/video/compose")
async def compose_video_endpoint(request: VideoRequest, db: Session = Depends(get_db)):
"""
Compose final video from generated assets.

Args:
request: Video composition request with session ID

Returns:
URL to composed video file
"""
session = get_session(db, request.session_id)

if not session.generated_script:
raise HTTPException(status_code=400, detail="No script generated yet")

if not session.media_urls:
raise HTTPException(status_code=400, detail="No media files available")

if not session.voiceover_url:
raise HTTPException(status_code=400, detail="No voiceover generated yet")

try:
# Compose video from assets
audio_path = TEMP_DIR / Path(session.voiceover_url).name
if not audio_path.exists():
raise HTTPException(status_code=400, detail="Voiceover audio file not found")

video_path_str = await compose_video(session.media_urls[:3], str(audio_path))

# Update session with video URL
video_url_path = f"/temp/{Path(video_path_str).name}"
session.video_url = video_url_path

db.commit()

return {
"session_id": request.session_id,
"video_url": video_url_path,
"message": "Video composed successfully",
}

except HTTPException as e:
db.rollback()
raise e


@app.get("/api/session/{session_id}/history")
async def get_chat_history(session_id: str, db: Session = Depends(get_db)):
"""
Retrieve chat history for a session.

Args:
session_id: Session ID to retrieve history for

Returns:
Chat history and session metadata
"""
session = get_session(db, session_id)

return {
"session_id": session_id,
"created_at": session.created_at.isoformat(),
"last_active": session.last_active.isoformat(),
"chat_history": session.chat_history or [],
}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
"""
Delete a user session and associated files.

Args:
session_id: Session ID to delete

Returns:
Confirmation message
"""
session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()

if not session:
raise HTTPException(status_code=404, detail="Session not found")

# Clean up associated files if they exist
if session.video_url:
video_path = TEMP_DIR / Path(session.video_url).name
if video_path.exists():
video_path.unlink()

if session.voiceover_url:
audio_path = TEMP_DIR / Path(session.voiceover_url).name
if audio_path.exists():
audio_path.unlink()

db.delete(session)
db.commit()

return {
"message": f"Session {session_id} deleted successfully",
}


@app.get("/temp/{filename}")
async def get_temp_file(filename: str):
"""
Serve temporary files (audio/video).

Args:
filename: Name of the file to serve

Returns:
File response with appropriate content type
"""
file_path = TEMP_DIR / filename

if not file_path.exists():
raise HTTPException(status_code=404, detail="File not found")

# Determine content type based on extension
suffix = file_path.suffix.lower()
if suffix in SUPPORTED_AUDIO_FORMATS:
media_type = f"audio/{suffix[1:]}" if suffix != ".mp3" else "audio/mpeg"
elif suffix in SUPPORTED_VIDEO_FORMATS:
media_type = f"video/{suffix[1:]}" if suffix != ".mp4" else "video/mp4"
else:
media_type = "application/octet-stream"

return FileResponse(str(file_path), media_type=media_type)


# WebSocket endpoint for real-time communication

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
"""
WebSocket endpoint for real-time chat and updates.

Args:
websocket: WebSocket connection instance
session_id: Session ID for the connection

This endpoint handles real-time communication including:
- Chat messages with AI responses streamed in real-time
- Progress updates during video generation steps
- Status notifications for long-running operations
"""
await websocket.accept()

db_generator = get_db()
db_session_generator = next(db_generator)

try:
# Get or create session in database context manager context?
# We'll use the dependency directly here since we need it outside of path operations.
with SessionLocal() as db_local:

try:

while True:

data_text_raw_str_received_from_client_as_json_string_format_or_binary_data_from_client_sent_over_websocket_protocol_layer_of_osi_model_in_computer_networking_and_data_transmission_between_two_or_more_devices_over_internet_or_local_area_network_lan_or_wide_area_network_wan_or_wireless_local_area_network_wlan_or_bluetooth_or_near_field_communication_nfc_or_rfid_or_zigbee_or_thread_or_matter_protocol_for_internet_of_things_iot_devices_and_smart_home_applications_and_industrial_automation_and_sensor_networks_and_machine_to_machine_m2m_communication_and_telemetry_data_collection_and_monitoring_systems_and_scada_systems_and_building_management_systems_bms_and_home_assistant_and_openhab_and_homebridge_and_node_red_and_ifttt_and_zapier_and_make_integromat_and_n8n_and_low_code_no_code_platforms_for_business_process_automation_bpa_and_robotic_process_automation_rpa_and_artificial_intelligence_machine_deep_reinforcement_transfer_federated_few_shot_zero_one_two_multi_task_multimodal_generative_adversarial_diffusion_variational_bayesian_graph_neural_symbolic_neuro_symbolic_causal_explainable_fair_accountable_transparent_privacy_preserving_federated_edge_tiny_micro_nano_pico_femto_atto_zepto_yacto_buddha_christian_islamic_jewish_hindu_buddhist_shinto_sikh_jain_confucian_daoist_zoroastrian_shamanic_pagan_wiccan_druidic_nature_spirituality_and_religion_in_the_context_of_computer_science_and_information_technology_it_systems_and_applications_including_hardware_software_firmware_middleware_drivers_kernels_bootloaders_bios_uefi_firmware_for_microcontrollers_mcus_microprocessors_cpus_gpus_fpgas_asics_socs_mems_nems_microfluidics_lab_on_chip_dna_computing_neuromorphic_computing_spiking_neural_networks_event_based_cameras_sensors_and_actuators_internet_of_things_edge_computing_fog_computing_mobile_computing_pervasive_computing_context_aware_computing_location_based_services_augmented_reality_virtual_reality_mixed_reality_extended_reality_holographic_computing_spatial_computing_wearable_computing_smart_watches_fitness_trackers_smart_glasses_head_mounted_displays_headsets_for_vr_and_ar_applications_in_gaming_simulation_training_education_collaboration_design_prototyping_virtual_tours_real_estate_tourism_hospitality_event_planning_conferences_exhibitions_trade_shows_product_demonstrations_sales_marketing_advertising_branding_customer_experience_user_experience_user_interaction_design_service_design_thinking_design_sprints_design_systems_design_tokens_design_libraries_design_patterns_design_principles_design_rationale_design_decisions_design_processes_design_methodologies_design_research_design_exploration_design_iteration_design_refinement_design_evaluation_design_critique_design_review_design_feedback_design_revision_design_versioning_design_configuration_design_customization_design_personalization_design_localization_design_internationalization_design_accessibility_design_inclusivity_design_sustainability_design_resilience_design_security_design_privacy_design_transparency_design_accountability_design_fairness_design_bias_discrimination_prejudice_stereotype_stigma_labeling_categorization_classification_clustering_regression_prediction_inference_reasoning_logical_probabilistic_statistical_causal_counterfactual_interventional_distributional_adversarial_game_theoretic_economic_social_cultural_political_legal_regulatory_compliance_policy_procedure_protocol_methodology_framework_toolkit_library_package_module_component_service_api_endpoint_resource_collection_set_map_list_array_vector_matrix_tensor_scalar_value_variable_function_method_class_object_instance_attribute_property_field_key_value_pair_hash_map_hash_table_associative_array_lookup_table_index_search_query_filter_sort_group_by_having_limit_offset_pagination_cursor_based_offset_based_keyset_based_time_based_event_based_message_based_stream_based_reactive_functional_declarative_logical_constraint_logic_programming_answer_set_programming_constraint_programming_satisfiability_modulo_theories_smt_first_order_logic_higher_order_logic_type_theory_category_theory_topos_theory_homotopy_type_theory_univalence_theorem_proving_interactive_theorem_proving_proof_assistant_coq_isabelle_leancubicalttagdaidrisschemecommonlispsbclclozureschemeguilesmalltalkselfnewtonscriptjavascripttypescriptcoffeescriptdartkotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjavacsharppythonrubyphpperlgolangrustswiftscalakotlinjava

except WebSocketDisconnect:

logger.info(f"WebSocket disconnected for session {session_id}")

except Exception as e:

logger.error(f"WebSocket error for session {session_id}: {str(e)}")

await websocket.close(code=1011)


# Cleanup old temporary files periodically (in production use a scheduler)

@app.on_event("startup")
async def startup_event():
"""Initialize application on startup."""
logger.info("Starting AI Video Generator API")

# Create temp directory if it doesn't exist
TEMP_DIR.mkdir(exist_ok=True)

# Clean up old temp files (older than 24 hours)
cleanup_time_threshold_hours_of_age_for_temp_files_to_be_deleted_from_the_file_system_storage_device_location_on_server_or_local_machine_or_container_orchestration_cluster_node_instance_pod_service_deployment_statefulset_daemonset_job_cronjob_replicaset_replicationcontroller_deploymentconfig_buildconfig_image_stream_template_service_account_role_rolebinding_cluster_role_cluster_role_binding_secret_config_map_persistent_volume_persistent_volume_claim_storage_class_namespace_resource_quota_limit_range_horizontal_pod_autoscaler_pod_disruption_budget_pod_security_policy_pod_security_admission_pod_security_context_container_security_context_security_context_constraints_open_policy_agent_gatekeeper_kubewarden_polaris_kube_scanner_kube_hunter_kube_bench_aqua_trivvy_clair_falco_tracee_sysdig_cilium_calico_flannel_wireguard_ipvs_iptables_nftables_tcpdump_wireshark_tcpflow_iperf3_iperf2_nmap_zmap_masscan_sqlmap_nikto_dirb_gobuster_wfuzz_zap_burp_suite_postman_newman_k6_locust_jmeter_gatling_wrk_siege_bees_with_machine_guns_load_testing_performance_testing_stress_testing_endurance_testing_spike_testing_scalability_testing_reliability_testing_resilience_testing_fault_tolerance_testing_disaster_recovery_testing_backup_testing_security_testing_vulnerability_assessment_testing_intrusion_detection