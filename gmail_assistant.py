import asyncio
import json
import os
import sys
import uuid
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

# Configuration
TOKENS_FILE = "gmail_tokens.json"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GMAIL_CLIENT_ID = os.environ.get("GMAIL_CLIENT_ID")
GMAIL_CLIENT_SECRET = os.environ.get("GMAIL_CLIENT_SECRET")

# Check for required environment variables
if not ANTHROPIC_API_KEY:
    print("Error: ANTHROPIC_API_KEY environment variable must be set")
    sys.exit(1)
if not GMAIL_CLIENT_ID or not GMAIL_CLIENT_SECRET:
    print("Error: GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET environment variables must be set")
    sys.exit(1)

# Initialize Anthropic client
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

class GmailMCPClient:
    def __init__(self, tokens_file: str = TOKENS_FILE):
        """Initialize the Gmail MCP client with OAuth tokens."""
        self.tokens = self._load_tokens(tokens_file)
        print("Tokens: ", self.tokens)
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.tools_cache = None

    def _load_tokens(self, tokens_file: str) -> Dict[str, Any]:
        """Load OAuth tokens from file."""
        try:
            with open(tokens_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Tokens file '{tokens_file}' not found.")
            print("Please create this file with your OAuth tokens in JSON format.")
            sys.exit(1)

    async def connect(self):
        """Connect to the Gmail MCP server."""
        server_params = StdioServerParameters(
            command="node",
            args=["dist/index.js"],
            env={
                "GMAIL_CLIENT_ID": GMAIL_CLIENT_ID,
                "GMAIL_CLIENT_SECRET": GMAIL_CLIENT_SECRET
            }
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()
        print("Connected to Gmail MCP server.")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools and cache them."""
        if self.tools_cache is not None:
            return self.tools_cache

        response = await self.session.list_tools()
        self.tools_cache = response.tools
        return self.tools_cache

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with provided arguments."""
        # Always add tokens to the arguments
        arguments["token"] = self.tokens

        print("Args: ", arguments)

        response = await self.session.call_tool(name, arguments)
        return response

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


class GmailAssistant:
    def __init__(self, gmail_client: GmailMCPClient):
        """Initialize the Gmail Assistant with a Gmail client."""
        self.gmail_client = gmail_client
        self.tools = None
        self.message_history = []
        self.system_prompt = """You are a Gmail assistant that helps users perform email operations.
            Your job is to understand user requests related to Gmail and determine the most appropriate tool to use.

            Guidelines:
            1. Extract all relevant parameters from the query (email addresses, subject lines, search terms, etc.)
            2. If any required information is missing, politely ask the user for the necessary details
            3. Use the most specific tool available for the task
            4. Format dates and times appropriately when applicable
            5. Make sure all required parameters for the selected tool are provided
            6. Explain clearly what actions you've taken and what the results mean
            7. When showing email search results, format them in a readable way

            Remember to use the available tools to help with email operations. Don't pretend to access Gmail directly."""

    async def _prepare_tools(self):
        """Prepare tools for Anthropic function calling."""
        # Get tools from MCP server
        mcp_tools = await self.gmail_client.list_tools()

        # Convert MCP tools to Anthropic tool format
        anthropic_tools = []
        for tool in mcp_tools:
            # Create Anthropic tool definition
            anthropic_tool = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            anthropic_tools.append(anthropic_tool)

        self.tools = anthropic_tools

        for tool in anthropic_tools:
            print("Tool: ", tool, " \n\n")
        return anthropic_tools

    async def process_query(self, query: str) -> str:
        """Process a user query using Anthropic Claude and available tools."""
        # Make sure tools are prepared
        if self.tools is None:
            await self._prepare_tools()

        # Add user query to message history
        self.message_history.append({
            "role": "user",
            "content": query
        })

        # Initial Claude API call with tools
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=self.message_history,
            system=self.system_prompt,
            tools=self.tools
        )

        # Process response and handle tool calls
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                # Add assistant's text response to history
                self.message_history.append({
                    "role": "assistant",
                    "content": content.text
                })
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                tool_id = str(uuid.uuid4())  # Generate unique ID for this tool use

                print(f"Calling tool: {tool_name}")
                print(f"With arguments: {tool_args}")

                # Execute tool call
                result = await self.gmail_client.call_tool(tool_name, tool_args)

                print("Tool call result - ", result)
                # Format the tool result into text
                result_text = ""
                for item in result.content:
                    if item.type == "text":
                        result_text += item.text

                final_text.append(f"[Tool result: {result_text}]")

                # Add tool use to history with the required ID field
                self.message_history.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,  # Add ID field
                            "name": tool_name,
                            "input": tool_args
                        }
                    ]
                })

                # Add tool result to history with matching ID
                self.message_history.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,  # Use the same ID
                            "content": result_text
                        }
                    ]
                })

                # Get next response from Claude
                follow_up_response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=self.message_history,
                    system=self.system_prompt,
                )

                # Add Claude's interpretation of tool results to output
                if follow_up_response.content and follow_up_response.content[0].type == 'text':
                    final_text.append(follow_up_response.content[0].text)
                    # Add this interpretation to history
                    self.message_history.append({
                        "role": "assistant",
                        "content": follow_up_response.content[0].text
                    })

        # Return the combined response
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive session with the Gmail Assistant."""
        print("\n===== Gmail AI Assistant (Claude) =====")
        print("Ask anything about your Gmail or request email operations.")
        print("Type 'exit' to quit.")

        # Initialize with empty message history
        self.message_history = []

        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in ['exit', 'quit', 'bye']:
                    break

                print("\nProcessing...")
                response = await self.process_query(query)
                print(f"\nAssistant: {response}")
            except Exception as e:
                print(f"\nError: {str(e)}")
                import traceback
                traceback.print_exc()


async def main():
    # Create clients
    gmail_client = GmailMCPClient()
    try:
        await gmail_client.connect()
        assistant = GmailAssistant(gmail_client)
        await assistant.chat_loop()
    finally:
        await gmail_client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())