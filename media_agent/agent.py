import os
import time

from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import FunctionTool, ToolContext, load_artifacts
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import Client, types
import uuid

MODEL = "gemini-2.5-flash"
MODEL_IMAGE = "imagen-4.0-fast-generate-preview-06-06"
MODEL_VIDEO = "veo-2.0-generate-001"

client = Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
)


async def generate_image(tool_context: "ToolContext", img_prompt: str, aspect_ratio: str = "16:9"):
    """Generates an image based on the prompt."""
    response = client.models.generate_images(
        model=MODEL_IMAGE,
        prompt=img_prompt,
        config={
            "number_of_images": 1,
            "aspect_ratio": aspect_ratio,
        },
    )
    if not response.generated_images:
        return {"status": "failed"}
    image_bytes = response.generated_images[0].image.image_bytes
    filename = f"generated_image_{uuid.uuid4()}.png" 
    version = await tool_context.save_artifact(
        filename,
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
    )
    print(f"Image saved with version: {version}")
    return {
        "status": "success",
        "detail": "Image generated successfully and stored in artifacts.",
        "filename": filename,
    }

async def generate_video(tool_context: "ToolContext", video_prompt: str, image_filename: str, aspect_ratio: str = "16:9"):
    """Generates a video based on the prompt and image."""
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
    

generate_image_tool = FunctionTool(func=generate_image)
generate_video_tool = FunctionTool(func=generate_video)
mcp_toolset = MCPToolset(
    connection_params=SseConnectionParams(url="http://localhost:8000/mcp"),
)

root_agent = Agent(
    name="media_agent",
    model=MODEL,
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
        )
    ),
    description=("Generalist agent able to generate images and videos."),
    instruction=(
        "You are a helpful agent who can answer user questions and generate images."
        "When user ask you to generate an image, always generate images using the generate_image tool."
        "When user ask you to generate a video, always generate an image first using the generate_image tool."
        "Then use the generate_video tool to generate a video from the image."
    ),
    tools=[generate_image_tool, generate_video_tool, load_artifacts, mcp_toolset],
)
