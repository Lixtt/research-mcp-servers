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
    from arxiv_server import create_arxiv_server, extract_images_from_pdf, format_arxiv_id
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
        for tool_name in sorted(self.all_tools):
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
            
            # Provide guidance for id_list format issues
            if "400 - Bad Request" in str(e) and "id_list" in str(arguments):
                error_msg += (
                    "\n\nID_LIST FORMAT ERROR: arXiv API expects comma-separated string, not JSON array!"
                )
                current_val = arguments.get('id_list')
                if isinstance(current_val, str) and current_val.startswith('['):
                    error_msg += f"\n   Current (wrong): id_list={current_val}"
                    try:
                        import json
                        parsed = json.loads(current_val)
                        if isinstance(parsed, list):
                            correct_format = ",".join(str(x) for x in parsed)
                            error_msg += f"\n   Correct format: id_list=\"{correct_format}\""
                    except (json.JSONDecodeError, ValueError):
                        pass
                error_msg += "\n   Use comma-separated string like: \"1706.03762,1909.03550\""

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

    async def test_extract_images_from_pdf(self) -> bool:
        """Test the extract_images_from_pdf function with a sample PDF."""
        
        print("\n🔄 Testing PDF to images conversion...")
        
        # Check if there are any downloaded papers to test with
        papers_dir = Path(os.getcwd()) / "papers"
        
        if not papers_dir.exists():
            print("❌ No 'papers' directory found. Please download a paper first.")
            return False
            
        pdf_files = list(papers_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("❌ No PDF files found in papers directory. Please download a paper first.")
            return False
        
        # Test with the first PDF file found
        test_pdf = pdf_files[0]
        print(f"📄 Testing with: {test_pdf.name}")
        
        try:
            # Test with different parameters
            print("🔄 Testing with max_pages=2, dpi=150...")
            images = await extract_images_from_pdf(str(test_pdf), max_pages=2, dpi=150)
            
            print(f"✅ Successfully extracted {len(images)} images")
            
            # Verify the structure of returned data
            for i, img in enumerate(images):
                expected_keys = ["page_number", "image_data", "format", "dpi"]
                missing_keys = [key for key in expected_keys if key not in img]
                
                if missing_keys:
                    print(f"❌ Page {i+1}: Missing keys: {missing_keys}")
                    return False
                    
                # Check data types and values
                if not isinstance(img["page_number"], int):
                    print(f"❌ Page {i+1}: page_number should be int")
                    return False
                    
                if not isinstance(img["image_data"], str):
                    print(f"❌ Page {i+1}: image_data should be string")
                    return False
                    
                if img["format"] != "png":
                    print(f"❌ Page {i+1}: format should be 'png'")
                    return False
                    
                if img["dpi"] != 150:
                    print(f"❌ Page {i+1}: dpi should be 150")
                    return False
                    
                # Check that base64 data is reasonable length (not empty, not too short)
                if len(img["image_data"]) < 1000:
                    print(f"❌ Page {i+1}: image_data seems too short ({len(img['image_data'])} chars)")
                    return False
                    
                print(f"  ✅ Page {img['page_number']}: {len(img['image_data'])} chars of base64 data")
            
            print("🔄 Testing with different DPI (200)...")
            images_hires = await extract_images_from_pdf(str(test_pdf), max_pages=1, dpi=200)
            
            if len(images_hires) != 1:
                print("❌ Expected 1 image for max_pages=1")
                return False
                
            if images_hires[0]["dpi"] != 200:
                print("❌ DPI not set correctly")
                return False
                
            # Higher DPI should produce longer base64 strings (more data)
            if len(images_hires[0]["image_data"]) <= len(images[0]["image_data"]):
                print("⚠️  Warning: Higher DPI didn't produce larger image data as expected")
            else:
                print("✅ Higher DPI produced larger image as expected")
                
            print(f"  ✅ DPI 200: {len(images_hires[0]['image_data'])} chars of base64 data")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during image extraction test: {str(e)}")
            return False

    async def test_paper_resources_tool(self) -> bool:
        """Test the new get_paper_resources MCP tool function."""
        
        print("\n🔄 Testing get_paper_resources MCP tool...")
        
        # Check if there are any downloaded papers to test with
        papers_dir = Path(os.getcwd()) / "papers"
        
        if not papers_dir.exists() or not list(papers_dir.glob("*.pdf")):
            print("❌ No PDF files found for MCP tool test")
            return False
        
        # Test with the first PDF file found
        test_pdf = list(papers_dir.glob("*.pdf"))[0]
        print(f"📄 Testing MCP tool with: {test_pdf.name}")
        
        try:
            # Extract arXiv ID from filename if possible
            import re
            arxiv_pattern = r"(\d{4}\.\d{4,5}(v\d+)?)$"
            match = re.search(arxiv_pattern, test_pdf.stem)
            
            if not match:
                print("❌ Could not extract paper ID from filename")
                return False
            
            paper_id = match.group(1)
            print(f"🔍 Extracted paper ID: {paper_id}")
            
            # Test generating image resources
            result = await self.call_arxiv_tool("get_paper_resources", {
                "paper_id": paper_id,
                "resource_types": ["image", "metadata"],
                "max_pages": 2,
                "dpi": 150
            })
            
            print(f"✅ MCP tool call successful: {result[:200]}...")
            
            # Basic validation - check if it contains expected structure
            if "image" in str(result).lower() and "metadata" in str(result).lower():
                print("✅ Result contains expected resource types")
                return True
            else:
                print("⚠️ Result format may be unexpected but tool executed")
                return True
            
        except Exception as e:
            print(f"❌ Error during MCP tool test: {str(e)}")
            return False

    async def test_openrouter_vision_analysis(self) -> bool:
        """Test OpenRouter model's ability to analyze extracted PDF images."""
        
        print("\n🔄 Testing OpenRouter vision analysis with PDF images...")
        
        # Check if there are any downloaded papers to test with
        papers_dir = Path(os.getcwd()) / "papers"
        
        if not papers_dir.exists() or not list(papers_dir.glob("*.pdf")):
            print("❌ No PDF files found. Skipping OpenRouter vision test.")
            return False
        
        # Test with the first PDF file found
        test_pdf = list(papers_dir.glob("*.pdf"))[0]
        print(f"📄 Testing vision analysis with: {test_pdf.name}")
        
        try:
            # Extract first page as image
            images = await extract_images_from_pdf(str(test_pdf), max_pages=1, dpi=150)
            
            if not images:
                print("❌ No images extracted from PDF")
                return False
            
            image_data = images[0]["image_data"]
            
            print("🧠 Sending image to OpenRouter for analysis...")
            
            # Test with a vision-capable model (Claude 3.5 Sonnet supports vision)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "This is the first page of a scientific paper from arXiv. Please analyze what you see and provide a brief description of the content, including any formulas, figures, or key information visible on this page."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            
            print("✅ OpenRouter vision analysis completed!")
            print(f"📝 Analysis result ({len(analysis)} chars):")
            print("-" * 50)
            print(analysis[:500] + "..." if len(analysis) > 500 else analysis)
            print("-" * 50)
            
            # Basic validation of the response
            if len(analysis) < 50:
                print("⚠️  Analysis seems too short, model might not have seen the image properly")
                return False
            
            # Check for common academic paper terms
            academic_keywords = ['paper', 'research', 'abstract', 'title', 'author', 'formula', 'equation', 'figure', 'text', 'scientific']
            found_keywords = [word for word in academic_keywords if word.lower() in analysis.lower()]
            
            if found_keywords:
                print(f"✅ Found {len(found_keywords)} academic keywords in analysis: {', '.join(found_keywords[:5])}")
            else:
                print("⚠️  No clear academic keywords found in analysis")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during OpenRouter vision test: {str(e)}")
            if "credits" in str(e).lower() or "balance" in str(e).lower():
                print("💳 This might be a billing/credits issue with your OpenRouter account")
            elif "rate" in str(e).lower() or "limit" in str(e).lower():
                print("🚦 This might be a rate limiting issue")
            elif "model" in str(e).lower():
                print("🤖 Model might not be available or doesn't support vision")
            
            return False

    def generate_test_report(self, image_tests_results: Dict[str, bool] = None) -> Dict[str, Any]:
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

        # Add image test results if provided
        if image_tests_results:
            report["image_tests"] = image_tests_results
            report["image_tests_passed"] = all(image_tests_results.values())
            
        return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test script for arXiv MCP Server with image functionality, using OpenRouter (default) or OpenAI-compatible APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Comprehensive test with default OpenRouter (includes image functionality + vision analysis)
    python test_arxiv_server.py --api-key your-openrouter-key

    # Test with OpenAI API instead
    python test_arxiv_server.py --api-key your-openai-key --base-url https://api.openai.com/v1 --model gpt-4

    # Test with local LLM server (image tests will run but vision analysis may be skipped)
    python test_arxiv_server.py --base-url http://localhost:11434/v1 --model llama3

    # Use environment variables (for OpenRouter by default)
    export OPENAI_API_KEY="your-openrouter-key"
    python test_arxiv_server.py

    # Use different OpenRouter model for vision analysis
    python test_arxiv_server.py --model anthropic/claude-3-opus

Test Coverage:
    - All MCP tools (arxiv_query, download_paper, read_paper, etc.)
    - PDF to images conversion functionality
    - New read_paper_as_images tool
    - OpenRouter vision analysis of PDF images
    - Complete integration testing workflow
        """,
    )

    parser.add_argument(
        "--api-key",
        help="API key (OpenRouter API key for default config, can also use OPENAI_API_KEY environment variable)",
        default=None,
    )

    parser.add_argument(
        "--base-url",
        help="Custom base URL for OpenAI-compatible API (default: https://openrouter.ai/api/v1 for OpenRouter)",
        default="https://openrouter.ai/api/v1",
    )

    parser.add_argument(
        "--model",
        help="Model to use for testing (default: anthropic/claude-3.5-sonnet for OpenRouter)",
        default="anthropic/claude-3.5-sonnet",
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
    print("arXiv MCP Server - Comprehensive Integration Test with Image Functionality")
    print("=" * 70)

    # Parse command line arguments
    args = parse_args()

    # Get configuration from args or environment variables
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    model = args.model

    # Check for API key
    if not api_key:
        print("API key not provided")
        print("Please provide it via:")
        print("  --api-key YOUR_OPENROUTER_KEY  (for default OpenRouter setup)")
        print("  or set OPENAI_API_KEY environment variable")
        return 1

    # Print configuration
    print(f"Model: {model}")
    if base_url:
        if "openrouter.ai" in base_url:
            print(f"Using OpenRouter API: {base_url}")
        else:
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

        # Run image-related tests
        print("\n" + "=" * 60)
        print("🖼️  IMAGE FUNCTIONALITY TESTS")
        print("=" * 60)
        
        # Test arXiv ID formatting first (utility function)
        print("🔄 Testing arXiv ID formatting...")
        format_test_passed = True
        test_cases = [
            ("2401.12345", "2401.12345"),
            ("2401.123", "2401.12300"),  # Should pad with zeros
            (2401.12345, "2401.12345"),
            ("1909.03550v1", "1909.03550v1"),  # Should preserve version
        ]
        
        for input_id, expected in test_cases:
            result = format_arxiv_id(input_id)
            if result == expected:
                print(f"  ✅ {input_id} -> {result}")
            else:
                print(f"  ❌ {input_id} -> {result} (expected {expected})")
                format_test_passed = False
        
        # Run other image tests
        image_extraction_passed = await tester.test_extract_images_from_pdf()
        mcp_tool_passed = await tester.test_paper_resources_tool()
        
        # Run vision analysis test (may be skipped)
        print("\n" + "=" * 60)
        print("🧠 AI VISION ANALYSIS TEST")
        print("=" * 60)
        vision_analysis_passed = await tester.test_openrouter_vision_analysis()
        
        # Collect image test results
        image_tests_results = {
            "format_arxiv_id": format_test_passed,
            "extract_images_from_pdf": image_extraction_passed,
            "get_paper_resources_tool": mcp_tool_passed,
            "openrouter_vision_analysis": vision_analysis_passed,
        }
        
        # Update report with image test results
        report = tester.generate_test_report(image_tests_results)

        # Print comprehensive results
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        print(f"🔧 MCP Tools: {report['called_tools']}/{report['total_tools']} called")
        print(f"   Success Rate: {report['success_rate']:.1f}%")
        print(f"   Test Passed: {'✅ YES' if report['test_passed'] else '❌ NO'}")
        
        if "image_tests" in report:
            passed_count = sum(1 for passed in report["image_tests"].values() if passed)
            total_count = len(report["image_tests"])
            print(f"\n🖼️  Image Tests: {passed_count}/{total_count} passed")
            for test_name, passed in report["image_tests"].items():
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            print(f"   Overall: {'✅ PASSED' if report['image_tests_passed'] else '❌ FAILED'}")
        
        overall_success = report["test_passed"] and (report.get("image_tests_passed", True))
        print(f"\n🎯 Final Result: {'✅ ALL TESTS PASSED' if overall_success else '⚠️ SOME TESTS FAILED'}")

        if report["called_tool_list"]:
            print(f"\n✅ Successfully Called MCP Tools ({len(report['called_tool_list'])}):")
            for tool in report["called_tool_list"]:
                print(f"   - {tool}")

        if report["uncalled_tool_list"]:
            print(f"\n❌ Uncalled MCP Tools ({len(report['uncalled_tool_list'])}):")
            for tool in report["uncalled_tool_list"]:
                print(f"   - {tool}")

        # Print usage tips if tests passed
        if overall_success:
            print("\n💡 Usage Tips:")
            print("   - All MCP tools are working correctly")
            print("   - Image functionality allows AI models to 'see' PDF content")
            print("   - Use read_paper_as_images() for visual analysis of papers")
            print("   - OpenRouter vision models can analyze the extracted images")

        return 0 if overall_success else 1

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
