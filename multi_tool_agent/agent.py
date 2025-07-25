import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from google.adk.agents import (
    Agent,
    LlmAgent,  # Any agent
)
from google.adk.agents.callback_context import CallbackContext
from google.adk.artifacts import InMemoryArtifactService  # Or GcsArtifactService
from google.adk.models import LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import agent_tool
from google.genai import types


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}

def get_landmarks(city: str) -> dict:
    """Returns the landmarks in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the landmarks.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        return {
            "status": "success",
            "landmarks": ["Statue of Liberty", "Empire State Building", "Central Park"],
        }
    else:
        return {
            "status": "error",
            "error_message": f"Landmarks information for '{city}' is not available.",
        }



def skip_summarization(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    print("callback_context")  
    print("--------------------------------")               
    print(callback_context.agent_name)
    print(callback_context.user_content)
    print("llm_request")
    print("--------------------------------")  
    print(llm_request.contents)
    print("--------------------------------")  
    response = None
    last_user_message = ''
    if llm_request.contents and llm_request.contents[-1].role == 'user':
        last_message = llm_request.contents[-1]
        if (
            last_message.parts
            and last_message.parts[0].function_response
        ):
            print("function_response skipping?")
            last_user_message = last_message.parts[0].function_response.response.get('result', '')
            if last_user_message:
                print("function_response skipping!")
                print(last_user_message)
                response = LlmResponse(
                    content=types.Content(
                        role='model',
                        parts=[types.Part(text=last_user_message)],
                    )
                )
            last_user_message = last_message.parts[0].function_response.response.get('landmarks', '')
            if last_user_message:
                print("function_response skipping!")
                print(last_user_message)
                response = LlmResponse(
                    content=types.Content(
                        role='model',
                        parts=[types.Part(text=', '.join(last_user_message))],
                    )
                )
    return response

landmarks_agent = Agent(
    name="landmarks_agent",
    model="gemini-2.0-flash",
    description="Agent to answer questions about landmarks in a city.",
    instruction="You are a helpful agent who can answer user questions about landmarks in a city.",
    # before_model_callback=skip_summarization,
    tools=[get_landmarks],
)

root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to answer questions about the time and weather in a city. You can use the landmarks_agent tool to get information about landmarks in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city."
    ),
    # before_model_callback=skip_summarization,
    tools=[get_weather, get_current_time, agent_tool.AgentTool(landmarks_agent)],
)