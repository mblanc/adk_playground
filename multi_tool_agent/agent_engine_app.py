from agent import root_agent
from vertexai.preview.reasoning_engines import AdkApp


def session_service_builder():
  from google.adk.sessions import InMemorySessionService

  return InMemorySessionService()

def artifact_service_builder():
  from google.adk.artifacts import InMemoryArtifactService

  return InMemoryArtifactService()

adk_app = AdkApp(
  agent=root_agent,
  enable_tracing=True,
  session_service_builder=session_service_builder,
  artifact_service_builder=artifact_service_builder,
  
)