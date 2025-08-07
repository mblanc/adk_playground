from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.planners import BuiltInPlanner

from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams,
    StdioServerParameters,
)
from google.genai import types
from pydantic import BaseModel

MODEL = "gemini-2.5-flash"
TARGET_FOLDER_PATH = (
    "/Users/matthieublanc/Projects/adk_playground/ai_news_agent/mcp_files"
)

playwright_mcp_tool = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@playwright/mcp@latest",
            ],
        ),
        timeout=20,
    ),
)


class Site(BaseModel):
    name: str
    url: str
    result_key: str


researcher_agents = []
sites = [
    Site(
        name="twitter",
        url="https://x.com/i/communities/1762494276565426592",
        result_key="twitter_result",
    ),
    Site(
        name="hacker_news",
        url="https://news.ycombinator.com/",
        result_key="hacker_news_result",
    ),
    Site(
        name="reddit",
        url="https://www.reddit.com/r/singularity/hot/",
        result_key="reddit_result",
    ),
]
for i, site in enumerate(sites):
    researcher_agent = Agent(
        name=f"researcher_{i + 1}",
        model="gemini-2.5-flash",
        instruction=(
            """Research AI news."""
            "In a new tab (use browser_tab_new)"
            "Navigate to this website, using browser_navigate, to get the latest news about AI and AI products and models:"
            f"{site.url}"
        ),
        tools=[playwright_mcp_tool],
        output_key=site.result_key,
    )
    researcher_agents.append(researcher_agent)

# ParallelAgent executes all researchers concurrently
parallel_research = ParallelAgent(
    name="ParallelToolExecution",
    sub_agents=researcher_agents,
    description="Executes all tool-using agents in parallel",
)

# Optional: Combine with SequentialAgent for post-processing
synthesis_agent = Agent(
    name="SynthesisAgent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a a specialist in AI and AI products and models."
        "Your goal is to generate news articles about AI and AI products and models."
        "Combine results from parallel research:"
    )
    + "\n".join([f"{site.name}: {site.result_key}" for site in sites])
    + ("Then generate a news article about the latest news."),
    description="Synthesizes parallel results",
)

# Full pipeline: parallel execution then synthesis
root_agent = SequentialAgent(
    name="ai_news_agent",
    description=(
        "AI news specialist agent that generates news articles about AI and AI products."
    ),
    sub_agents=[parallel_research, synthesis_agent],
)

# root_agent = Agent(
#     name="ai_news_agent",
#     model=MODEL,
#     planner=BuiltInPlanner(
#         thinking_config=types.ThinkingConfig(
#             include_thoughts=True,
#         )
#     ),
#     description=(
#         "AI news specialist agent that generates news articles about AI and AI products."
#     ),
#     instruction=(
#         "You are a a specialist in AI and AI products and models."
#         "Your goal is to generate news articles about AI and AI products and models."
#         "First use browser_navigate to get the latest news about AI and AI products and models using all these pages: "
#         "https://x.com/i/communities/1762494276565426592"
#         "You don't need to login here, you can just use the page as is to grab the latest news from this community"
#         "https://news.ycombinator.com/"
#         "https://www.reddit.com/r/singularity/hot/"
#         "https://techcrunch.com/category/artificial-intelligence/"
#         "https://www.theverge.com/ai-artificial-intelligence"
#         "https://openai.com/news/"
#         "https://aiweekly.co/"
#         "https://www.artificialintelligence-news.com/"
#         "https://aibusiness.com/ml"
#         "https://venturebeat.com/"
#         "https://www.technologyreview.com/topic/artificial-intelligence/"
#         "https://www.sciencedaily.com/news/computers_math/artificial_intelligence/"
#         "https://www.wired.com/tag/artificial-intelligence/"
#         "https://www.forbes.com/ai/"
#         "https://blog.google/technology/ai/"
#         "https://cloud.google.com/blog/products/ai-machine-learning"
#         "https://deepmind.google/discover/blog/"
#         "Make sure to visit all these pages to get the latest news."
#         "Then extract the news from these pages and generate a news article about the latest news. "
#     ),
#     tools=[
#         MCPToolset(
#             connection_params=StdioConnectionParams(
#                 server_params=StdioServerParameters(
#                     command="npx",
#                     args=[
#                         "-y",  # Argument for npx to auto-confirm install
#                         "@playwright/mcp@latest",
#                     ],
#                 ),
#                 timeout=20,
#             ),
#             # Optional: Filter which tools from the MCP server are exposed
#             # tool_filter=['list_directory', 'read_file']
#         )
#     ],
# )
