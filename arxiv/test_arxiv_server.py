import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

from openai import AsyncOpenAI


# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from arxiv_server import create_arxiv_server
    from fastmcp import FastMCP, Client
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)


class ArxivServerManager:
    """Manages the arXiv server lifecycle for testing."""

    def __init__(self):
        self.server = None

    def start_server(self):
        """Initialize the arXiv server."""
        print("Initializing arXiv server...")
        self.server = create_arxiv_server()
        print("Server is ready!")

    async def get_server_tools(self) -> Dict[str, Any]:
        """Get all available tools from the server."""
        try:
            if not self.server:
                self.server = create_arxiv_server()
            return await self.server.get_tools()
        except Exception as e:
            print(f"Failed to get server tools: {e}")
            return {}

    def get_server(self) -> FastMCP:
        """Get the server instance."""
        if not self.server:
            self.server = create_arxiv_server()
        return self.server

    def stop_server(self):
        """Stop the server."""
        print("Server manager cleaned up...")


class OpenAIFunctionConverter:
    """Converts FastMCP tools to OpenAI function calling format."""

    @staticmethod
    def convert_tool_to_function(tool_name: str, tool_info) -> Dict[str, Any]:
        """Convert a FastMCP tool to OpenAI function format using server-provided schema."""
        try:
            # Get the tool's description from the server
            description = getattr(tool_info, "description", f"Tool: {tool_name}")

            # Get parameter schema from FastMCP tool
            parameters = {"type": "object", "properties": {}, "required": []}

            # Try different ways to get the schema
            if hasattr(tool_info, "parameters") and tool_info.parameters:
                # Use the parameters attribute directly
                parameters = tool_info.parameters
            elif hasattr(tool_info, "to_mcp_tool"):
                # Get schema from MCP tool
                mcp_tool = tool_info.to_mcp_tool()
                if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
                    parameters = mcp_tool.inputSchema

            # Ensure we have a valid schema structure
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}, "required": []}
            if not parameters.get("properties"):
                parameters["properties"] = {}
            if "type" not in parameters:
                parameters["type"] = "object"

            function_def = {
                "name": tool_name,
                "description": description,
                "parameters": parameters,
            }

            return function_def

        except Exception as e:
            print(f"Failed to convert tool {tool_name}: {e}")
            # Return a basic definition as fallback
            return {
                "name": tool_name,
                "description": f"Tool: {tool_name}",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }


class OpenAITester:
    """Tests arXiv server using OpenAI function calling."""

    def __init__(
        self,
        server_manager: ArxivServerManager,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-5",
    ):
        self.server_manager = server_manager
        self.model = model

        # Initialize OpenAI client with custom base_url if provided
        client_kwargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncOpenAI(**client_kwargs)
        self.converter = OpenAIFunctionConverter()
        self.called_tools = set()
        self.all_tools = set()
        self.error_history = []  # Track repeated errors

    async def prepare_tools(self) -> List[Dict[str, Any]]:
        """Prepare all arXiv tools as OpenAI tools using server-provided schemas."""
        server_tools = await self.server_manager.get_server_tools()
        self.all_tools = set(server_tools.keys())

        tools = []
        for tool_name, tool_info in server_tools.items():
            function_def = self.converter.convert_tool_to_function(tool_name, tool_info)
            tools.append({"type": "function", "function": function_def})

            # Debug: Print tool information to verify schema extraction
            description = getattr(tool_info, "description", "No description")
            print(f"Tool '{tool_name}': {description[:100]}...")

            # Show parameter count to verify schema was extracted
            param_count = len(function_def.get("parameters", {}).get("properties", {}))
            print(f"   Parameters: {param_count}")

        print(f"Prepared {len(tools)} tools for OpenAI from server:")
        for tool_name in self.all_tools:
            print(f"   - {tool_name}")

        return tools

    async def call_arxiv_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an arXiv tool via FastMCP Client."""
        try:
            # Get the server instance
            server = self.server_manager.get_server()

            # Use FastMCP Client to call the tool
            async with Client(server) as client:
                result = await client.call_tool(tool_name, arguments)

                # Track that this tool was called
                self.called_tools.add(tool_name)

                # Extract content from the result
                if hasattr(result, "content") and result.content:
                    if isinstance(result.content, list) and len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, "text"):
                            return content_item.text
                        elif isinstance(content_item, str):
                            return content_item

                # If we can't extract text, return the string representation
                return str(result)

        except Exception as e:
            error_msg = f"Error calling {tool_name}: {str(e)}"
            print(f"{error_msg}")

            # Track this error for repeated error detection
            error_signature = f"{tool_name}:{str(e)}"
            self.error_history.append(error_signature)

            # Check for repeated errors (same error 3+ times in a row)
            if len(self.error_history) >= 3:
                recent_errors = self.error_history[-3:]
                if all(err == recent_errors[0] for err in recent_errors):
                    # Same error repeated 3 times - add strong guidance
                    error_msg += "\n\nREPEATED ERROR DETECTED! You've made the same mistake 3 times."

            # Provide specific guidance for common type errors
            if "is not of type 'string'" in str(e) and "paper_id" in str(arguments):
                error_msg += (
                    "\n\nCRITICAL FIX NEEDED: paper_id must be a STRING with quotes!"
                )
                error_msg += (
                    f"\n   Current (wrong): paper_id={arguments.get('paper_id')}"
                )
                error_msg += (
                    f"\n   Correct format: paper_id=\"{arguments.get('paper_id')}\""
                )
                error_msg += (
                    "\n   Always use quotes around paper IDs in function calls!"
                )

            return error_msg

    def get_initial_prompt(self) -> str:
        """Get the initial prompt to guide the model."""
        return """You are a helpful assistant testing an arXiv research paper server.

I need you to systematically test ALL available tools by calling them with appropriate parameters.

Start by exploring the arXiv query functionality first, then proceed to test other available tools like downloading and reading papers.

Please make function calls to test each tool and demonstrate their capabilities."""

    def get_continuation_prompt(self, uncalled_tools: List[str]) -> str:
        """Get a prompt to encourage calling remaining tools."""
        return f"""Great work so far! You still need to test these remaining tools: {', '.join(uncalled_tools)}

Please continue testing by making function calls to the remaining tools. Each tool has its own parameters and functionality - use the function definitions provided to understand how to call them properly."""

    async def run_comprehensive_test(self, max_rounds: int = 15) -> Dict[str, Any]:
        """Run a comprehensive test of all tools using OpenAI."""
        print("Starting comprehensive OpenAI integration test...")

        # Prepare tools
        tools = await self.prepare_tools()

        # Initialize conversation
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can call functions to test an arXiv paper server.",
            },
            {"role": "user", "content": self.get_initial_prompt()},
        ]

        round_count = 0

        while round_count < max_rounds:
            round_count += 1
            print(f"\nRound {round_count}")

            try:
                # Call OpenAI with tool calling
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3,
                )

                message = response.choices[0].message

                # Prepare assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": message.content,
                }

                # Add tool calls if they exist
                if message.tool_calls:
                    assistant_message["tool_calls"] = [
                        tool_call.model_dump() for tool_call in message.tool_calls
                    ]

                messages.append(assistant_message)

                # Check if the model wants to call tools
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        print(f"Calling tool: {function_name}")
                        print(f"Arguments: {json.dumps(function_args, indent=2)}")

                        # Check for potential repeated errors before calling
                        error_signature_preview = (
                            f"{function_name}:Input validation error"
                        )
                        if len(self.error_history) >= 2:
                            recent_errors = self.error_history[-2:]
                            if all(
                                error_signature_preview in err for err in recent_errors
                            ):
                                print(
                                    "Detected potential repeated error pattern - adding extra guidance"
                                )

                        # Call the actual arXiv tool
                        result = await self.call_arxiv_tool(
                            function_name, function_args
                        )

                        # Add tool result to conversation
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )

                        print(f"Tool {function_name} completed")

                        # If we have repeated errors, force guidance
                        if len(self.error_history) >= 3:
                            recent_errors = self.error_history[-3:]
                            if all(err == recent_errors[0] for err in recent_errors):
                                print(
                                    "BREAKING ERROR LOOP - Adding corrective guidance"
                                )
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": "You are making the same error repeatedly. STOP using numbers for paper_id. Use STRINGS with quotes. Try a different tool or approach now.",
                                    }
                                )

                else:
                    # Model didn't call a tool, check if we need to prompt for more
                    uncalled_tools = list(self.all_tools - self.called_tools)

                    if uncalled_tools:
                        print(f"Model response: {message.content}")
                        print(f"Prompting to call remaining tools: {uncalled_tools}")

                        # Add a prompt to call remaining tools
                        messages.append(
                            {
                                "role": "user",
                                "content": self.get_continuation_prompt(uncalled_tools),
                            }
                        )
                    else:
                        print("All tools have been tested!")
                        break

            except Exception as e:
                print(f"Error in round {round_count}: {e}")
                import traceback

                traceback.print_exc()
                break

        # Generate test report
        return self.generate_test_report()

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        uncalled_tools = list(self.all_tools - self.called_tools)

        report = {
            "total_tools": len(self.all_tools),
            "called_tools": len(self.called_tools),
            "success_rate": (
                len(self.called_tools) / len(self.all_tools) * 100
                if self.all_tools
                else 0
            ),
            "called_tool_list": sorted(list(self.called_tools)),
            "uncalled_tool_list": sorted(uncalled_tools),
            "test_passed": len(uncalled_tools) == 0,
        }

        return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test script for arXiv MCP Server using OpenAI SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default OpenAI API
    python test_openai_integration.py

    # Use custom base URL (e.g., local LLM or other provider)
    python test_openai_integration.py --base-url http://localhost:11434/v1

    # Use specific model
    python test_openai_integration.py --model gpt-5

    # Use environment variables
    export OPENAI_API_KEY="your-key"
    export OPENAI_BASE_URL="http://your-api-endpoint/v1"
    python test_openai_integration.py
        """,
    )

    parser.add_argument(
        "--api-key",
        help="OpenAI API key (can also use OPENAI_API_KEY environment variable)",
        default=None,
    )

    parser.add_argument(
        "--base-url",
        help="Custom base URL for OpenAI-compatible API (can also use OPENAI_BASE_URL environment variable)",
        default=None,
    )

    parser.add_argument(
        "--model",
        help="Model to use for testing (default: gpt-5)",
        default="gpt-5",
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        help="Maximum number of conversation rounds (default: 15)",
        default=15,
    )

    return parser.parse_args()


async def main():
    """Main test function."""
    print("arXiv MCP Server - OpenAI Integration Test")
    print("=" * 60)

    # Parse command line arguments
    args = parse_args()

    # Get configuration from args or environment variables
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    model = args.model

    # Check for OpenAI API key
    if not api_key:
        print("OpenAI API key not provided")
        print("Please provide it via:")
        print("  --api-key YOUR_KEY")
        print("  or set OPENAI_API_KEY environment variable")
        return 1

    # Print configuration
    print(f"Model: {model}")
    if base_url:
        print(f"Using custom base URL: {base_url}")
    else:
        print("Using default OpenAI API")

    server_manager = None

    try:
        # Initialize the arXiv server
        server_manager = ArxivServerManager()
        server_manager.start_server()

        # Initialize OpenAI tester
        tester = OpenAITester(
            server_manager, api_key=api_key, base_url=base_url, model=model
        )

        # Run comprehensive test
        report = await tester.run_comprehensive_test(max_rounds=args.max_rounds)

        # Print results
        print("\n" + "=" * 60)
        print("TEST REPORT")
        print("=" * 60)
        print(f"Total Tools: {report['total_tools']}")
        print(f"Tools Called: {report['called_tools']}")
        print(f"Success Rate: {report['success_rate']:.1f}%")
        print(f"Test Passed: {'YES' if report['test_passed'] else 'NO'}")

        if report["called_tool_list"]:
            print(f"\nSuccessfully Called Tools ({len(report['called_tool_list'])}):")
            for tool in report["called_tool_list"]:
                print(f"   - {tool}")

        if report["uncalled_tool_list"]:
            print(f"\nUncalled Tools ({len(report['uncalled_tool_list'])}):")
            for tool in report["uncalled_tool_list"]:
                print(f"   - {tool}")

        return 0 if report["test_passed"] else 1

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if server_manager:
            server_manager.stop_server()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
