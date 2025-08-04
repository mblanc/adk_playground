import vertexai
from vertexai import agent_engines
from vertexai.preview import reasoning_engines
from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext
from google.genai import Client, types
from pydantic import BaseModel
from typing import List
from typing import Optional
import uuid
import time


MODEL = "gemini-2.5-flash"
MODEL_IMAGE = "imagen-4.0-fast-generate-preview-06-06"
MODEL_VIDEO = "veo-2.0-generate-001"

PROJECT_ID = "svc-demo-vertex"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://svc-demo-vertex-us"

class AgentSpec(BaseModel):
    name: str
    model: str
    description: str
    instruction: str
    tools: Optional[List[str]] = None
    sub_agents: Optional[List[str]] = None


async def generate_image(tool_context: "ToolContext", img_prompt: str, aspect_ratio: str = "16:9"):
    """Generates an image based on the prompt."""
    client = Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )
    print("#"*27)
    print(img_prompt)
    response = client.models.generate_images(
        model=MODEL_IMAGE,
        prompt=img_prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1, 
            aspect_ratio=aspect_ratio, 
            enhance_prompt=True),
    )
    if not response.generated_images:
        return {"status": "failed"}
    image_bytes = response.generated_images[0].image.image_bytes
    filename = f"generated_image_{uuid.uuid4()}.png" 
    version = await tool_context.save_artifact(
        filename,
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
    )
    prompt = img_prompt
    if response.generated_images[0].enhanced_prompt:
        prompt = response.generated_images[0].enhanced_prompt
    tool_context.state.update({filename : prompt})
    print(f"Image saved with version: {version}")
    return {
        "status": "success",
        "detail": "Image generated successfully and stored in artifacts.",
        "filename": filename,
    }

async def modify_image(tool_context: "ToolContext", img_prompt: str, image_filename: str, aspect_ratio: str = "16:9"):
    """Modify an image based on the modified prompt."""
    client = Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )
    response = client.models.generate_images(
        model=MODEL_IMAGE,
        prompt=img_prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1, 
            aspect_ratio=aspect_ratio, 
            enhance_prompt=True),
    )
    if not response.generated_images:
        return {"status": "failed"}
    image_bytes = response.generated_images[0].image.image_bytes
    version = await tool_context.save_artifact(
        image_filename,
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
    )
    prompt = img_prompt
    if response.generated_images[0].enhanced_prompt:
        prompt = response.generated_images[0].enhanced_prompt
    tool_context.state.update({image_filename : prompt})
    print(f"Image saved with version: {version}")
    return {
        "status": "success",
        "detail": "Image generated successfully and stored in artifacts.",
        "filename": image_filename,
    }

async def generate_video(tool_context: "ToolContext", video_prompt: str, image_filename: str, aspect_ratio: str = "16:9"):
    """Generates a video based on the prompt and image."""
    client = Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )
    image_artifact = await tool_context.load_artifact(image_filename)
    if image_artifact and image_artifact.inline_data:
        print(f"Successfully loaded latest Python artifact '{image_filename}'.")
        print(f"MIME Type: {image_artifact.inline_data.mime_type}")
        # Process the report_artifact.inline_data.data (bytes)
        image_bytes = image_artifact.inline_data.data
        print(f"Report size: {len(image_bytes)} bytes.")
        # ... further processing ...
        operation = client.models.generate_videos(
            model=MODEL_VIDEO,
            prompt=video_prompt,
            image=types.Image(image_bytes=image_bytes, mime_type="image/png"),
            config={
                "number_of_videos": 1,
                "aspect_ratio": aspect_ratio,
            },
        )

        while not operation.done:
            time.sleep(5)
            operation = client.operations.get(operation)

        if not operation.result.generated_videos:
            return {"status": "failed"}
        video_bytes = operation.result.generated_videos[0].video.video_bytes
        filename = f"generated_video_{uuid.uuid4()}.mp4" 
        version = await tool_context.save_artifact(
            filename,
            types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
        )
        print(f"Video saved with version: {version}")
        return {
            "status": "success",
            "detail": "Video generated successfully and stored in artifacts.",
            "filename": filename,
        }
    else:
        print(f"Python artifact '{image_filename}' not found.")
        return {
            "status": "error",
            "detail": "Image not found."
        }


tool_map = {
    "generate_image": FunctionTool(func=generate_image),
    "generate_video": FunctionTool(func=generate_video)
}

def create_agent_from_spec(agent_spec: AgentSpec) -> Agent:
    agent = Agent(
        name=agent_spec.name,
        model=agent_spec.model,
        description=agent_spec.description,
        instruction=agent_spec.instruction,
        tools=[tool_map[tool] for tool in agent_spec.tools],
    )
    return agent



 
print("I am fine")
 
if __name__ == "__main__":
    agent_spec = {
        "name": "",
    }

    agent_spec = AgentSpec(
        name="media_agent",
        model="gemini-2.5-flash",
        description=("Media agent able to generate images and videos."),
        instruction=(
            "You are a helpful agent who can answer user questions and generate images."
            "When user ask you to generate an image, always generate images using the generate_image tool."
            "When user ask you to generate a video, always generate an image first using the generate_image tool."
            "Then use the generate_video tool to generate a video from the image."
        ),
        tools=["generate_image", "generate_video"],
    )

    print("Create an agent dynamically")
    agent = create_agent_from_spec(agent_spec)
    print(agent)
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=STAGING_BUCKET,
    )


    print("Create ADK app")
    app = reasoning_engines.AdkApp(
        agent=agent,
        enable_tracing=True,
        session_service_builder=None,
        artifact_service_builder=None,
        memory_service_builder=None,
        env_vars={},
    )
    print(app)

    print("Deploy to Agent Engine")
    remote_app = agent_engines.create(
        agent_engine=app,
        requirements=[
            "google-cloud-aiplatform[adk,agent_engines]",
        ],
        display_name="Dynamic agent",
        description="Dynamic agent creation test",
        gcs_dir_name=None,
        extra_packages=None,
        env_vars=None,
        build_options=None,
    )
    print(remote_app)