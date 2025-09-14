import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import aiofiles
import httpx
from pydantic import BaseModel, Field

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Set environment variables to avoid X11 issues in headless environments
os.environ["DISPLAY"] = ""
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["NO_AT_BRIDGE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Constants
ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
ARXIV_ABS_BASE = "https://arxiv.org/abs"
DEFAULT_MAX_RESULTS = 10
DEFAULT_START = 0


# Create the server object
mcp = FastMCP("arXiv Papers Server")

# HTTP client for arXiv API
arxiv_client = httpx.AsyncClient(timeout=60.0)


class ArxivPaper(BaseModel):
    """Represents an arXiv paper."""

    id: str = Field(description="arXiv paper ID (e.g., 2401.12345)")
    title: str = Field(description="Paper title")
    authors: list[str] = Field(description="List of authors")
    abstract: str = Field(description="Paper abstract")
    published: datetime = Field(description="Publication date")
    updated: datetime = Field(description="Last update date")
    categories: list[str] = Field(description="arXiv categories")
    pdf_url: str = Field(description="PDF download URL")
    abs_url: str = Field(description="Abstract page URL")
    doi: str | None = Field(default=None, description="DOI if available")
    comment: str | None = Field(default=None, description="Author comments")
    journal_ref: str | None = Field(default=None, description="Journal reference")


class SearchParameters(BaseModel):
    """Parameters for arXiv paper search - supports both search_query and id_list modes."""

    # Query parameters (mutually exclusive with id_list)
    search_query: str | None = Field(default=None, description="Search query string")
    id_list: str | list[str] | None = Field(
        default=None,
        description="Comma-separated list of arXiv IDs or list of arXiv IDs (strings)",
    )

    # Common parameters
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS, description="Maximum number of results"
    )
    start: int = Field(default=DEFAULT_START, description="Start index for pagination")
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Field(
        default="relevance", description="Sort order"
    )
    sort_order: Literal["ascending", "descending"] = Field(
        default="descending", description="Sort direction"
    )

    def model_post_init(self, __context):
        """Validate that search_query and id_list are mutually exclusive and convert list to string."""
        # Convert list id_list to comma-separated string
        if self.id_list and isinstance(self.id_list, list):
            self.id_list = ",".join(format_arxiv_id(id_val) for id_val in self.id_list)
        # Handle case where id_list is a JSON string representation of a list
        elif self.id_list and isinstance(self.id_list, str) and self.id_list.startswith('[') and self.id_list.endswith(']'):
            try:
                import json
                parsed_list = json.loads(self.id_list)
                if isinstance(parsed_list, list):
                    self.id_list = ",".join(format_arxiv_id(id_val) for id_val in parsed_list)
            except json.JSONDecodeError:
                # If it's not valid JSON, treat it as a regular string
                pass

        if self.search_query and self.id_list:
            raise ValueError("search_query and id_list cannot be used simultaneously")
        if not self.search_query and not self.id_list:
            raise ValueError("Either search_query or id_list must be provided")


def format_arxiv_id(paper_id: str | int | float) -> str:
    """
    Safely format arXiv paper ID to string, preserving format.

    Args:
        paper_id: arXiv paper ID (string, int, or float)

    Returns:
        Properly formatted arXiv ID string
    """
    if isinstance(paper_id, str):
        # Handle string input - ensure proper format
        paper_id = paper_id.strip()
        if "." in paper_id:
            parts = paper_id.split(".")
            if len(parts) == 2 and len(parts[0]) == 4 and parts[1].isdigit():
                # Ensure at least 5 digits after decimal for arXiv format
                if len(parts[1]) < 5:
                    paper_id = f"{parts[0]}.{parts[1].ljust(5, '0')}"
        return paper_id
    elif isinstance(paper_id, float):
        # Convert float to string with high precision then format
        formatted = f"{paper_id:.5f}".rstrip("0").rstrip(".")

        # Ensure we have proper arXiv format (YYMM.NNNNN)
        if "." in formatted:
            parts = formatted.split(".")
            if len(parts) == 2 and len(parts[0]) == 4 and parts[1].isdigit():
                # Ensure at least 5 digits after decimal for arXiv format
                if len(parts[1]) < 5:
                    formatted = f"{parts[0]}.{parts[1].ljust(5, '0')}"
        return formatted
    else:
        return str(paper_id)


async def parse_arxiv_response(xml_content: str) -> list[ArxivPaper]:
    """Parse arXiv API Atom feed response."""
    import xml.etree.ElementTree as ET

    papers = []
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    try:
        root = ET.fromstring(xml_content)

        for entry in root.findall("atom:entry", namespace):
            # Extract paper ID safely
            id_elem = entry.find("atom:id", namespace)
            if id_elem is None or id_elem.text is None:
                continue
            paper_id = id_elem.text.split("/")[-1]

            # Keep version number as returned by arXiv API

            # Extract title
            title_elem = entry.find("atom:title", namespace)
            title = (
                title_elem.text.strip()
                if title_elem is not None and title_elem.text
                else "No title"
            )

            # Extract authors
            authors = []
            for author_elem in entry.findall("atom:author/atom:name", namespace):
                if author_elem.text:
                    authors.append(author_elem.text.strip())

            # Extract abstract
            abstract_elem = entry.find("atom:summary", namespace)
            abstract = (
                abstract_elem.text.strip()
                if abstract_elem is not None and abstract_elem.text
                else "No abstract"
            )

            # Extract dates safely
            published_elem = entry.find("atom:published", namespace)
            updated_elem = entry.find("atom:updated", namespace)

            published = datetime.now()
            if published_elem is not None and published_elem.text:
                published = datetime.fromisoformat(
                    published_elem.text.replace("Z", "+00:00")
                )

            updated = datetime.now()
            if updated_elem is not None and updated_elem.text:
                updated = datetime.fromisoformat(
                    updated_elem.text.replace("Z", "+00:00")
                )

            # Extract categories
            categories = []
            for category_elem in entry.findall("atom:category", namespace):
                if "term" in category_elem.attrib:
                    categories.append(category_elem.attrib["term"])

            # Extract DOI and comments
            doi = None
            comment = None
            journal_ref = None

            for link_elem in entry.findall("atom:link", namespace):
                if link_elem.attrib.get("title") == "doi":
                    doi = (
                        link_elem.attrib.get("href", "").split("doi.org/")[-1]
                        if "doi.org/" in link_elem.attrib.get("href", "")
                        else None
                    )

            # Try to find comment and journal reference
            for elem in entry:
                if "comment" in elem.tag.lower() and elem.text:
                    comment = elem.text.strip()
                elif "journal_ref" in elem.tag.lower() and elem.text:
                    journal_ref = elem.text.strip()

            paper = ArxivPaper(
                id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                updated=updated,
                categories=categories,
                pdf_url=f"{ARXIV_PDF_BASE}/{paper_id}",
                abs_url=f"{ARXIV_ABS_BASE}/{paper_id}",
                doi=doi,
                comment=comment,
                journal_ref=journal_ref,
            )
            papers.append(paper)

    except ET.ParseError as e:
        raise ToolError(f"Failed to parse arXiv XML response: {str(e)}")

    return papers


async def search_arxiv_papers(params: SearchParameters) -> list[ArxivPaper]:
    """Search for papers on arXiv using the API - supports both search_query and id_list modes."""

    # Build query parameters based on the mode
    if params.search_query:
        # Search query mode
        query_params = {
            "search_query": params.search_query,
            "max_results": params.max_results,
            "start": params.start,
            "sortBy": params.sort_by,
            "sortOrder": params.sort_order,
        }
    else:
        # ID list mode
        query_params = {
            "id_list": params.id_list,
            "max_results": params.max_results,
            "start": params.start,
            "sortBy": params.sort_by,
            "sortOrder": params.sort_order,
        }

    try:
        response = await arxiv_client.get(ARXIV_API_BASE, params=query_params)
        response.raise_for_status()

        # Parse Atom feed response
        papers = await parse_arxiv_response(response.text)
        return papers

    except httpx.HTTPStatusError as e:
        raise ToolError(
            f"arXiv API error: {e.response.status_code} - {e.response.reason_phrase}"
        )
    except httpx.RequestError as e:
        raise ToolError(f"Network error: {str(e)}")
    except Exception as e:
        raise ToolError(f"Error parsing arXiv response: {str(e)}")


async def download_paper_pdf(
    paper_id: str, paper_title: str, save_directory: str | None = None
) -> dict:
    """
    Download a paper PDF from arXiv with automatic path creation.

    Args:
        paper_id: arXiv paper ID (e.g., "2401.12345")
        paper_title: Paper title for filename generation
        save_directory: Optional directory path to save PDF. If not provided,
                       uses default "papers" directory.

    Returns:
        Dictionary containing download result with file information
    """
    import re

    paper_id_str = format_arxiv_id(paper_id)
    # For PDF download, arXiv URLs work with or without version numbers
    # Keep the original ID format as provided by user
    clean_paper_id = paper_id_str

    # Clean title for filename
    clean_title = re.sub(r"[^\w\s-]", "", paper_title)
    clean_title = re.sub(r"[-\s]+", "-", clean_title)
    clean_title = clean_title.strip("-")

    # Set up save directory and file path
    if save_directory:
        target_dir = save_directory
    else:
        target_dir = os.path.join(os.getcwd(), "papers")

    # Ensure directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Create filename and full path
    # Limit title length to avoid filesystem issues
    safe_title = clean_title[:80] if clean_title else "paper"
    filename = f"{safe_title}_{paper_id_str}.pdf"
    save_path = os.path.join(target_dir, filename)

    # Download the PDF
    pdf_url = f"{ARXIV_PDF_BASE}/{clean_paper_id}"

    try:
        response = await arxiv_client.get(pdf_url)
        response.raise_for_status()

        async with aiofiles.open(save_path, "wb") as f:
            await f.write(response.content)

        return {
            "paper_id": paper_id_str,
            "title": paper_title,
            "filename": filename,
            "file_path": save_path,
            "status": "success",
            "message": f"PDF saved to: {save_path}",
        }

    except httpx.HTTPStatusError as e:
        raise ToolError(
            f"Failed to download PDF: {e.response.status_code} - {e.response.reason_phrase}"
        )
    except Exception as e:
        raise ToolError(f"Error downloading paper: {str(e)}")


async def extract_text_from_pdf(pdf_path: str, max_pages: int = 10) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract (default: 10)

    Returns:
        Extracted text content
    """
    try:
        import fitz  # PyMuPDF

        text_content = []
        with fitz.open(pdf_path) as doc:
            # Extract text from each page, up to max_pages
            for page_num in range(min(len(doc), max_pages)):
                page = doc.load_page(page_num)
                text = page.get_text()
                text_content.append(f"--- Page {page_num + 1} ---\n\n{text}")

        return "\n\n\n\n".join(text_content)

    except ImportError:
        raise ToolError(
            "PyMuPDF is required for PDF text extraction. Please install it with: pip install PyMuPDF"
        )
    except Exception as e:
        raise ToolError(f"Error reading PDF file: {str(e)}")


async def extract_images_from_pdf(pdf_path: str, max_pages: int = 10, dpi: int = 150) -> list[dict]:
    """
    Extract pages from a PDF file as base64-encoded images.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract (default: 10)
        dpi: Resolution for image conversion (default: 150)

    Returns:
        List of dictionaries containing page number and base64-encoded image data
    """
    try:
        import fitz  # PyMuPDF
        import base64

        images = []
        with fitz.open(pdf_path) as doc:
            # Extract images from each page, up to max_pages
            for page_num in range(min(len(doc), max_pages)):
                page = doc.load_page(page_num)
                
                # Convert page to image (PNG format)
                # The matrix determines the resolution - dpi/72 gives us the scaling factor
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PNG bytes
                img_data = pix.pil_tobytes("PNG")
                
                # Encode to base64
                img_b64 = base64.b64encode(img_data).decode('utf-8')
                
                images.append({
                    "page_number": page_num + 1,
                    "image_data": img_b64,
                    "format": "png",
                    "dpi": dpi
                })

        return images

    except ImportError:
        raise ToolError(
            "PyMuPDF is required for PDF image extraction. Please install it with: pip install PyMuPDF"
        )
    except Exception as e:
        raise ToolError(f"Error extracting images from PDF file: {str(e)}")


# Tools - Native arXiv API forwarding
@mcp.tool
async def arxiv_query(
    search_query: str = "",
    id_list: str | list[str] = "",
    max_results: int = DEFAULT_MAX_RESULTS,
    start: int = DEFAULT_START,
    sortBy: str = "relevance",
    sortOrder: str = "descending",
) -> list[dict]:
    """
    Native arXiv API query interface - direct forwarding with complete parameter support.

    Provides two query modes (mutually exclusive):
    1. Search query mode: Using search_query parameter with native arXiv syntax
    2. ID list mode: Using id_list parameter to fetch specific papers by ID

    Args:
        search_query: arXiv search query using native syntax. Supports:
                     - Basic: "machine learning", "neural networks"
                     - Author: "au:Hinton", "au:Geoffrey AND au:Hinton"
                     - Category: "cat:cs.AI", "cat:cs.LG OR cat:cs.CL"
                     - Title: "ti:transformer", "ti:attention mechanism"
                     - Abstract: "abs:deep learning"
                     - ID: "id:2401.12345"
                     - Date range: "submittedDate:[202301010000 TO 202312312359]"
                     - Complex: "cat:cs.AI AND au:Bengio AND submittedDate:[202301010000 TO 202312312359]"
        id_list: Comma-separated list of arXiv IDs or list of arXiv IDs (alternative to search_query, mutually exclusive)
                String example: "2401.12345,1909.03550,2312.11805"
                List of strings: ["2401.12345", "1909.03550", "2312.11805"]
        max_results: Maximum results to return (1-2000, default: 10)
        start: Start index for pagination (default: 0)
        sortBy: Sort field - "relevance", "lastUpdatedDate", or "submittedDate"
        sortOrder: Sort direction - "ascending" or "descending"

    Returns:
        List of paper objects with complete arXiv metadata

    Search Query Examples:
        arxiv_query("machine learning transformers")
        arxiv_query("cat:cs.AI AND submittedDate:[202301010000 TO 202312312359]")
        arxiv_query("au:Hinton AND submittedDate:[202301010000 TO 202401010000]")

    ID List Examples:
        arxiv_query(id_list="2401.12345,1909.03550")
        arxiv_query(id_list=["2401.12345", "1909.03550"])

    Version Number Examples:
        arxiv_query(id_list="1706.03762")      # Gets latest version
        arxiv_query(id_list="1706.03762v1")    # Gets specific version v1
        arxiv_query(id_list="1706.03762v5")    # Gets specific version v5

    Note:
    - search_query and id_list cannot be used simultaneously.
    - Version numbers are preserved as specified by the user.
    - Omitting version number returns the latest version of the paper.
    """
    # Convert list id_list to comma-separated string if needed
    if isinstance(id_list, list):
        id_list = ",".join(format_arxiv_id(id_val) for id_val in id_list)
    # Handle case where id_list is a JSON string representation of a list
    elif isinstance(id_list, str) and id_list.startswith('[') and id_list.endswith(']'):
        try:
            import json
            parsed_list = json.loads(id_list)
            if isinstance(parsed_list, list):
                id_list = ",".join(format_arxiv_id(id_val) for id_val in parsed_list)
        except json.JSONDecodeError:
            # If it's not valid JSON, treat it as a regular string
            pass

    # Build SearchParameters based on the query type
    if id_list:
        # Use ID list query mode
        params = SearchParameters(
            id_list=id_list,
            max_results=max_results,
            start=start,
            sort_by=sortBy,
            sort_order=sortOrder,
        )
    else:
        # Use search query mode
        params = SearchParameters(
            search_query=search_query if search_query else "all",
            max_results=max_results,
            start=start,
            sort_by=sortBy,
            sort_order=sortOrder,
        )

    # Use the centralized search function
    papers = await search_arxiv_papers(params)
    return [paper.model_dump() for paper in papers]


@mcp.tool
async def download_paper(
    paper_id: str | list[str],
    save_directory: str | None = None,
) -> str | dict:
    """
    Download one or more paper PDFs from arXiv to local storage.

    Args:
        paper_id: Single arXiv paper ID or list of paper IDs
                 Single: "2401.12345", "1909.03550"
                 List: ["2401.12345", "1909.03550"]
                 Version numbers are preserved as specified by user
        save_directory: Optional directory path to save PDF(s). If not provided,
                       uses default "papers" directory.

    Returns:
        For single paper: Success message with file path where PDF was saved
        For multiple papers: Dictionary with download results for each paper

    Examples:
        # Single paper download
        download_paper("2401.12345")        # Latest version
        download_paper("2401.12345v1")      # Specific version v1
        download_paper("1909.03550", "/home/user/papers")

        # Batch download
        download_paper(["2401.12345", "1909.03550", "2312.11805"])
        download_paper(["1706.03762v1", "1706.03762v5"])  # Different versions
        download_paper(["2401.12345", "1909.03550"], "/home/user/papers")
    """
    # Normalize input to always work with a list
    if isinstance(paper_id, list):
        paper_ids = paper_id
        is_batch = True
    else:
        paper_ids = [paper_id]
        is_batch = False

    # Handle case where paper_id might be a JSON string representation
    processed_ids = []
    for pid in paper_ids:
        if isinstance(pid, str) and pid.startswith('[') and pid.endswith(']'):
            try:
                import json
                parsed_list = json.loads(pid)
                if isinstance(parsed_list, list):
                    processed_ids.extend(parsed_list)
                else:
                    processed_ids.append(pid)
            except json.JSONDecodeError:
                processed_ids.append(pid)
        else:
            processed_ids.append(pid)
    
    # Convert all IDs to strings and remove duplicates while preserving order
    paper_id_strs = [format_arxiv_id(pid) for pid in processed_ids]
    seen = set()
    unique_ids = []
    for pid in paper_id_strs:
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)

    # For batch processing, prepare results dictionary
    if is_batch:
        results = {
            "total_requested": len(paper_ids),
            "total_unique": len(unique_ids),
            "successful_downloads": 0,
            "failed_downloads": 0,
            "downloads": [],
            "errors": [],
        }

    # Get metadata for all papers efficiently using batch query
    paper_map = {}
    if len(unique_ids) > 1:
        # Use batch query for multiple papers
        try:
            # Keep original IDs with version numbers as specified by user
            params = SearchParameters(
                id_list=unique_ids, max_results=len(unique_ids), start=0
            )
            papers = await search_arxiv_papers(params)
            paper_map = {paper.id: paper for paper in papers}
        except Exception as e:
            if is_batch:
                results["errors"].append(f"Failed to fetch batch metadata: {str(e)}")

    # Process each paper
    for paper_id_str in unique_ids:
        try:
            # Get paper title and info
            if paper_id_str in paper_map:
                paper = paper_map[paper_id_str]
                title = paper.title
            else:
                # Individual query for this paper
                try:
                    params = SearchParameters(
                        id_list=paper_id_str, max_results=1, start=0
                    )
                    individual_papers = await search_arxiv_papers(params)
                    if individual_papers:
                        title = individual_papers[0].title
                    else:
                        error_msg = f"Paper {paper_id_str} not found"
                        if is_batch:
                            results["errors"].append(error_msg)
                            results["downloads"].append(
                                {
                                    "paper_id": paper_id_str,
                                    "status": "failed",
                                    "error": error_msg,
                                }
                            )
                            results["failed_downloads"] += 1
                            continue
                        else:
                            raise ToolError(error_msg)
                except Exception as e:
                    error_msg = f"Failed to get info for {paper_id_str}: {str(e)}"
                    if is_batch:
                        results["errors"].append(error_msg)
                        results["downloads"].append(
                            {
                                "paper_id": paper_id_str,
                                "status": "failed",
                                "error": str(e),
                            }
                        )
                        results["failed_downloads"] += 1
                        continue
                    else:
                        raise ToolError(error_msg)

            # Download the paper using the unified download function
            download_result = await download_paper_pdf(
                paper_id_str, title, save_directory
            )

            # Record result
            if is_batch:
                results["downloads"].append(download_result)
                results["successful_downloads"] += 1
            else:
                # For single paper, return the message directly
                return download_result["message"]

        except Exception as e:
            error_msg = f"Failed to download {paper_id_str}: {str(e)}"
            if is_batch:
                results["errors"].append(error_msg)
                results["downloads"].append(
                    {"paper_id": paper_id_str, "status": "failed", "error": str(e)}
                )
                results["failed_downloads"] += 1
            else:
                raise ToolError(error_msg)

    # For batch processing, add summary and return results
    if is_batch:
        if results["successful_downloads"] == len(unique_ids):
            results["summary"] = (
                f"Successfully downloaded all {results['successful_downloads']} papers"
            )
        elif results["successful_downloads"] > 0:
            results["summary"] = (
                f"Downloaded {results['successful_downloads']} out of {len(unique_ids)} papers. "
                f"{results['failed_downloads']} failed."
            )
        else:
            results["summary"] = (
                f"Failed to download any papers. All {len(unique_ids)} downloads failed."
            )

        return results


@mcp.tool
async def list_downloaded_papers(
    directory: str | None = None,
    include_content_preview: bool = False,
    max_preview_chars: int = 1000,
) -> list[dict]:
    """
    List all downloaded arXiv papers in local storage.

    Args:
        directory: Directory to scan for PDFs. If not provided, uses default "papers" directory.
        include_content_preview: Whether to include text preview of paper content
        max_preview_chars: Maximum characters for content preview

    Returns:
        List of paper information including filename, filepath, title, arXiv ID, file size, download date

    Examples:
        list_downloaded_papers()
        list_downloaded_papers("/path/to/papers", include_content_preview=True)
    """

    # Determine directory to scan
    if directory and directory != "null" and directory.lower() != "none":
        scan_dir = Path(directory)
    else:
        scan_dir = Path(os.getcwd()) / "papers"

    # Ensure directory exists
    if not scan_dir.exists():
        return []

    # Find all PDF files
    pdf_files = list(scan_dir.glob("*.pdf"))

    papers = []

    for pdf_file in pdf_files:
        # Extract metadata from filename (format: title_arxiv_id.pdf)
        filename = pdf_file.stem

        # Try to extract arXiv ID from filename
        arxiv_id = None
        title = filename

        # Look for arXiv ID pattern in filename
        import re

        arxiv_pattern = r"(\d{4}\.\d{4,5}(v\d+)?)$"
        match = re.search(arxiv_pattern, filename)

        if match:
            arxiv_id = match.group(1)
            # Remove the arXiv ID from title
            title = (
                filename[: match.start()].rstrip("_- ")
                if match.start() > 0
                else "Unknown Title"
            )

        # Get file stats
        stat = pdf_file.stat()
        file_size_mb = stat.st_size / (1024 * 1024)
        file_size_kb = stat.st_size / 1024

        # Prepare paper info
        paper_info = {
            "filename": pdf_file.name,
            "filepath": str(pdf_file),
            "title": title,
            "arxiv_id": arxiv_id,
            "size_bytes": stat.st_size,
            "size_mb": round(file_size_mb, 2),
            "size_kb": round(file_size_kb, 2),
            "download_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "download_date_human": datetime.fromtimestamp(stat.st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "file_type": "pdf",
            "readable": True,
        }

        # Add content preview if requested
        if include_content_preview and arxiv_id:
            try:
                # Extract first page content for preview
                preview_text = await extract_text_from_pdf(str(pdf_file), max_pages=1)
                if preview_text:
                    # Clean up the preview text
                    preview_text = re.sub(r"\s+", " ", preview_text.strip())
                    if len(preview_text) > max_preview_chars:
                        preview_text = preview_text[:max_preview_chars] + "..."
                    paper_info["content_preview"] = preview_text
            except Exception:
                paper_info["content_preview"] = "Unable to extract preview"
                paper_info["readable"] = False

        papers.append(paper_info)

    # Sort by download date (newest first)
    papers.sort(key=lambda x: x["download_date"], reverse=True)

    return papers


@mcp.tool
async def read_paper(
    filepath: str = "",
    paper_id: str = "",
    max_pages: int = 10,
    max_chars: int = 100000,
) -> str:
    """
    Read and extract text content from a downloaded arXiv paper PDF.

    Provide either filepath OR paper_id to locate the file.

    Args:
        filepath: Direct path to PDF file to read
        paper_id: arXiv paper ID to automatically locate downloaded file (string)
        max_pages: Maximum pages to extract (default: 10)
        max_chars: Maximum characters to return (default: 100000)

    Returns:
        Extracted text content from PDF with file information

    Examples:
        read_paper(filepath="/path/to/paper.pdf")
        read_paper(paper_id="2401.12345", max_pages=15, max_chars=150000)
    """
    target_filepath = ""

    if filepath:
        # Read by direct file path
        target_filepath = filepath
    elif paper_id:
        # Convert paper_id to string if it's a number
        paper_id_str = format_arxiv_id(paper_id)

        # Find file by paper ID - scan directory directly
        scan_dir = Path(os.getcwd()) / "papers"
        if not scan_dir.exists():
            raise ToolError(
                f"No downloaded paper found with ID: {paper_id_str}. "
                f"Use download_paper first to download the paper."
            )

        # Find all PDF files
        pdf_files = list(scan_dir.glob("*.pdf"))
        matching_files = []

        # Look for files with matching arXiv ID pattern - improved search
        import re

        arxiv_pattern = r"(\d{4}\.\d{4,5}(v\d+)?)"

        for pdf_file in pdf_files:
            filename = pdf_file.stem
            # Try multiple patterns to find the arXiv ID in filename
            matches = re.findall(arxiv_pattern, filename)
            for match in matches:
                file_id = match[0]  # match[0] contains the full ID with optional version
                # Support both exact match and base ID match (without version)
                exact_match = file_id == paper_id_str
                base_match = file_id.split("v")[0] == paper_id_str.split("v")[0]
                if exact_match or base_match:
                    matching_files.append(str(pdf_file))
                    break

        if not matching_files:
            # Try to find by filename pattern containing the ID
            for pdf_file in pdf_files:
                if paper_id_str in pdf_file.name or paper_id_str.split("v")[0] in pdf_file.name:
                    matching_files.append(str(pdf_file))

        if not matching_files:
            raise ToolError(
                f"No downloaded paper found with ID: {paper_id_str}. "
                f"Use download_paper first to download the paper."
            )

        # Use the first matching paper
        target_filepath = matching_files[0]
    else:
        raise ToolError("Either filepath or paper_id must be provided")

    # Check if file exists
    if not os.path.exists(target_filepath):
        raise ToolError(f"File not found: {target_filepath}")

    # Check if it's a PDF file
    if not target_filepath.lower().endswith(".pdf"):
        raise ToolError("Only PDF files are supported for reading")

    try:
        # Extract text from PDF
        full_text = await extract_text_from_pdf(target_filepath, max_pages)

        # Truncate if exceeds max_chars
        if len(full_text) > max_chars:
            full_text = (
                f"{full_text[:max_chars]}"
                f"\n\n[Text truncated to {max_chars} characters]"
            )

        # Get file info for context
        file_stat = os.stat(target_filepath)
        file_size_mb = file_stat.st_size / (1024 * 1024)

        return (
            f"Paper Content (from {os.path.basename(target_filepath)}, "
            f"{file_size_mb:.1f} MB):\n\n{full_text}"
        )

    except Exception as e:
        raise ToolError(f"Error reading paper: {str(e)}")


# Note: read_paper_as_images is now implemented as MCP resources instead of a tool
# Use get_paper_image_resources tool to get resource URIs, then reference them directly

async def _extract_paper_images_internal(
    filepath: str = "",
    paper_id: str = "",
    max_pages: int = 5,
    dpi: int = 150,
) -> list[dict]:
    """
    Read arXiv paper PDF and convert pages to base64-encoded images for visual analysis by AI models.
    
    This function provides an alternative to text extraction by converting PDF pages to images,
    which allows AI models to see the paper's visual layout, formulas, figures, and formatting.

    Provide either filepath OR paper_id to locate the file.

    Args:
        filepath: Direct path to PDF file to read
        paper_id: arXiv paper ID to automatically locate downloaded file (string)
        max_pages: Maximum pages to convert to images (default: 5, recommended for performance)
        dpi: Image resolution in dots per inch (default: 150, higher = better quality but larger files)

    Returns:
        List of dictionaries with page images in base64 format:
        [
            {
                "page_number": 1,
                "image_data": "base64_encoded_png_data",
                "format": "png",
                "dpi": 150
            },
            ...
        ]

    Examples:
        read_paper_as_images(filepath="/path/to/paper.pdf", max_pages=3)
        read_paper_as_images(paper_id="2401.12345", max_pages=10, dpi=200)

    Note:
        - The returned images can be directly processed by AI models for visual analysis
        - Higher DPI values produce better quality but larger image files
        - Use max_pages wisely as images can be large - start with fewer pages
        - PNG format preserves text clarity and mathematical formulas
    """
    target_filepath = ""

    if filepath:
        # Read by direct file path
        target_filepath = filepath
    elif paper_id:
        # Convert paper_id to string if it's a number
        paper_id_str = format_arxiv_id(paper_id)

        # Find file by paper ID - scan directory directly
        scan_dir = Path(os.getcwd()) / "papers"
        if not scan_dir.exists():
            raise ToolError(
                f"No downloaded paper found with ID: {paper_id_str}. "
                f"Use download_paper first to download the paper."
            )

        # Find all PDF files
        pdf_files = list(scan_dir.glob("*.pdf"))
        matching_files = []

        # Look for files with matching arXiv ID pattern - improved search
        import re

        arxiv_pattern = r"(\d{4}\.\d{4,5}(v\d+)?)"

        for pdf_file in pdf_files:
            filename = pdf_file.stem
            # Try multiple patterns to find the arXiv ID in filename
            matches = re.findall(arxiv_pattern, filename)
            for match in matches:
                file_id = match[0]  # match[0] contains the full ID with optional version
                # Support both exact match and base ID match (without version)
                exact_match = file_id == paper_id_str
                base_match = file_id.split("v")[0] == paper_id_str.split("v")[0]
                if exact_match or base_match:
                    matching_files.append(str(pdf_file))
                    break

        if not matching_files:
            # Try to find by filename pattern containing the ID
            for pdf_file in pdf_files:
                if paper_id_str in pdf_file.name or paper_id_str.split("v")[0] in pdf_file.name:
                    matching_files.append(str(pdf_file))

        if not matching_files:
            raise ToolError(
                f"No downloaded paper found with ID: {paper_id_str}. "
                f"Use download_paper first to download the paper."
            )

        # Use the first matching paper
        target_filepath = matching_files[0]
    else:
        raise ToolError("Either filepath or paper_id must be provided")

    # Check if file exists
    if not os.path.exists(target_filepath):
        raise ToolError(f"File not found: {target_filepath}")

    # Check if it's a PDF file
    if not target_filepath.lower().endswith(".pdf"):
        raise ToolError("Only PDF files are supported for image extraction")

    try:
        # Extract images from PDF
        images = await extract_images_from_pdf(target_filepath, max_pages, dpi)
        
        # Add metadata to the response
        file_stat = os.stat(target_filepath)
        file_size_mb = file_stat.st_size / (1024 * 1024)
        
        # Add file information to each image and format for MCP client
        formatted_images = []
        for img in images:
            img["source_file"] = os.path.basename(target_filepath)
            img["file_size_mb"] = round(file_size_mb, 1)
            img["total_pages_in_pdf"] = len(images)
            
            # Create a data URI for better MCP client compatibility
            img["image_url"] = f"data:image/png;base64,{img['image_data']}"
            
            formatted_images.append(img)

        # Return the raw image data for internal use
        return formatted_images

    except Exception as e:
        raise ToolError(f"Error extracting images from paper: {str(e)}")


# MCP Resources for paper images - this is the main way to access paper images
@mcp.resource("arxiv://{paper_id}/{resource_type}/{resource_path}")
async def get_arxiv_resource(paper_id: str, resource_type: str, resource_path: str):
    """
    Provide arXiv resources including paper images, metadata, and content.
    
    Supported URI formats:
    - arxiv://{paper_id}/image/{page_number}?dpi={dpi}  - Get paper page as image
    - arxiv://{paper_id}/metadata  - Get paper metadata
    - arxiv://{paper_id}/text?pages={max_pages}  - Get paper text content
    
    Examples:
    - arxiv://1706.03762/image/1?dpi=150
    - arxiv://1706.03762/metadata
    - arxiv://1706.03762/text?pages=5
    """
    try:
        # Construct the full URI for reference
        uri = f"arxiv://{paper_id}/{resource_type}/{resource_path}"
        
        if resource_type == "image":
            # Parse page number and optional DPI from resource_path
            if "?" in resource_path:
                page_info, params = resource_path.split("?", 1)
                page_number = int(page_info)
                
                # Parse DPI parameter
                dpi = 150  # default
                for param in params.split("&"):
                    if param.startswith("dpi="):
                        dpi = int(param.split("=")[1])
            else:
                page_number = int(resource_path)
                dpi = 150
            
            return await _get_paper_image_resource(paper_id, page_number, dpi, uri)
            
        elif resource_type == "metadata":
            return await _get_paper_metadata_resource(paper_id, uri)
            
        elif resource_type == "text":
            # Parse optional pages parameter from resource_path
            max_pages = 10  # default
            if "?" in resource_path:
                params = resource_path.split("?", 1)[1]
                for param in params.split("&"):
                    if param.startswith("pages="):
                        max_pages = int(param.split("=")[1])
            
            return await _get_paper_text_resource(paper_id, max_pages, uri)
            
        else:
            raise ValueError(f"Unsupported resource type: {resource_type}")
            
    except Exception as e:
        raise ToolError(f"Error providing arXiv resource: {str(e)}")


async def _get_paper_image_resource(paper_id: str, page_number: int, dpi: int, uri: str):
    """Get a specific page of a paper as an image resource."""
    # Extract the specific page as image
    images = await _extract_paper_images_internal(
        paper_id=paper_id,
        max_pages=page_number,  # Extract up to the requested page
        dpi=dpi
    )
    
    if not images:
        raise ValueError(f"No images found for paper {paper_id}")
    
    # Find the specific page
    target_image = None
    for img in images:
        if img["page_number"] == page_number:
            target_image = img
            break
    
    if not target_image:
        raise ValueError(f"Page {page_number} not found in paper {paper_id}")
    
    # Return as MCP Resource dictionary
    return {
        "uri": uri,
        "name": f"arXiv {paper_id} - Page {page_number}",
        "description": f"Page {page_number} of arXiv paper {paper_id} as PNG image (DPI: {dpi})",
        "mimeType": "image/png",
        "text": f"Visual content of page {page_number} from arXiv paper {paper_id}. Contains figures, equations, and formatted text that can be analyzed by vision-capable AI models.",
        "blob": target_image["image_data"]  # base64 encoded image data
    }


async def _get_paper_metadata_resource(paper_id: str, uri: str):
    """Get metadata for a paper as a resource."""
    try:
        # Query arXiv for metadata using the global function
        papers = await arxiv_query(id_list=paper_id, max_results=1)
        
        if not papers:
            raise ValueError(f"Paper {paper_id} not found")
        
        paper = papers[0]
        
        # Format metadata as readable text
        metadata_text = f"""arXiv Paper Metadata: {paper_id}

Title: {paper['title']}

Authors: {', '.join(paper['authors'])}

Abstract: {paper['abstract']}

Categories: {', '.join(paper['categories'])}

Published: {paper['published']}
Updated: {paper['updated']}

PDF URL: {paper['pdf_url']}
Abstract URL: {paper['abs_url']}
"""
        
        if paper.get('doi'):
            metadata_text += f"DOI: {paper['doi']}\n"
        if paper.get('journal_ref'):
            metadata_text += f"Journal Reference: {paper['journal_ref']}\n"
        if paper.get('comment'):
            metadata_text += f"Comments: {paper['comment']}\n"
        
        return {
            "uri": uri,
            "name": f"arXiv {paper_id} - Metadata",
            "description": f"Complete metadata for arXiv paper {paper_id}",
            "mimeType": "text/plain",
            "text": metadata_text
        }
        
    except Exception as e:
        raise ValueError(f"Error getting metadata for {paper_id}: {str(e)}")


async def _get_paper_text_resource(paper_id: str, max_pages: int, uri: str):
    """Get text content of a paper as a resource."""
    try:
        # Read paper text content
        content = await read_paper(paper_id=paper_id, max_pages=max_pages, max_chars=100000)
        
        return {
            "uri": uri,
            "name": f"arXiv {paper_id} - Text Content",
            "description": f"Text content of arXiv paper {paper_id} (first {max_pages} pages)",
            "mimeType": "text/plain",
            "text": content
        }
        
    except Exception as e:
        raise ValueError(f"Error getting text content for {paper_id}: {str(e)}")


# Legacy resource handler for backward compatibility
@mcp.resource("paper-image://{paper_id}/{page_info}")
async def get_paper_image_resource_legacy(paper_id: str, page_info: str):
    """Legacy resource handler for paper-image:// URIs. Use arxiv:// instead."""
    try:
        # Construct the full URI for reference
        uri = f"paper-image://{paper_id}/{page_info}"
        
        # Parse page number and optional DPI
        if "?" in page_info:
            page_number, params = page_info.split("?", 1)
            page_number = int(page_number)
            
            # Parse DPI parameter
            dpi = 150  # default
            for param in params.split("&"):
                if param.startswith("dpi="):
                    dpi = int(param.split("=")[1])
        else:
            page_number = int(page_info)
            dpi = 150
        
        return await _get_paper_image_resource(paper_id, page_number, dpi, uri)
        
    except Exception as e:
        raise ToolError(f"Error providing image resource: {str(e)}")


@mcp.tool
async def get_paper_resources(
    paper_id: str,
    resource_types: list[str] = ["image"],
    max_pages: int = 3,
    dpi: int = 150
) -> dict:
    """
    Generate MCP resource URIs for arXiv papers that can be directly referenced by AI models.
    
    This is the main way to access arXiv paper content as MCP resources. The generated URIs
    can be referenced in conversations and will be automatically loaded by compatible MCP clients.
    
    Args:
        paper_id: arXiv paper ID (e.g., "1706.03762")
        resource_types: Types of resources to generate (default: ["image"])
                       Options: "image", "metadata", "text"
        max_pages: For image/text resources, maximum number of pages (default: 3)
        dpi: For image resources, resolution (default: 150)
    
    Returns:
        Dictionary with resource URIs organized by type
    
    Examples:
        # Get image resources
        resources = await get_paper_resources("1706.03762", ["image"], max_pages=2)
        # Returns: {"images": [{"uri": "arxiv://1706.03762/image/1?dpi=150", ...}]}
        
        # Get all resource types
        resources = await get_paper_resources("1706.03762", ["image", "metadata", "text"])
        
    Usage in conversation:
        "Please analyze this paper's first page: arxiv://1706.03762/image/1"
        "Show me the metadata: arxiv://1706.03762/metadata"
    """
    paper_id_str = format_arxiv_id(paper_id)
    
    # For metadata, we don't need local files - can query arXiv directly
    need_local_files = any(rt in resource_types for rt in ["image", "text"])
    
    if need_local_files:
        # Verify the paper exists locally
        scan_dir = Path(os.getcwd()) / "papers"
        if not scan_dir.exists():
            raise ToolError(f"No papers directory found. Download paper {paper_id_str} first.")
        
        # Find the paper file
        pdf_files = list(scan_dir.glob("*.pdf"))
        matching_files = []
        
        import re
        arxiv_pattern = r"(\d{4}\.\d{4,5}(v\d+)?)"
        
        for pdf_file in pdf_files:
            filename = pdf_file.stem
            matches = re.findall(arxiv_pattern, filename)
            for match in matches:
                file_id = match[0]
                exact_match = file_id == paper_id_str
                base_match = file_id.split("v")[0] == paper_id_str.split("v")[0]
                if exact_match or base_match:
                    matching_files.append(str(pdf_file))
                    break
        
        if not matching_files:
            raise ToolError(f"Paper {paper_id_str} not found locally. Use download_paper first.")
    
    # Generate resources by type
    result = {}
    
    if "image" in resource_types:
        images = []
        for page_num in range(1, max_pages + 1):
            uri = f"arxiv://{paper_id_str}/image/{page_num}?dpi={dpi}"
            images.append({
                "uri": uri,
                "page_number": page_num,
                "paper_id": paper_id_str,
                "dpi": dpi,
                "resource_type": "image",
                "description": f"Page {page_num} of arXiv paper {paper_id_str} as image",
                "usage": "Reference this URI in conversation - MCP clients will show the image"
            })
        result["images"] = images
    
    if "metadata" in resource_types:
        metadata_uri = f"arxiv://{paper_id_str}/metadata"
        result["metadata"] = {
            "uri": metadata_uri,
            "paper_id": paper_id_str,
            "resource_type": "metadata",
            "description": f"Complete metadata for arXiv paper {paper_id_str}",
            "usage": "Reference this URI to get paper title, authors, abstract, etc."
        }
    
    if "text" in resource_types:
        text_uri = f"arxiv://{paper_id_str}/text?pages={max_pages}"
        result["text"] = {
            "uri": text_uri,
            "paper_id": paper_id_str,
            "max_pages": max_pages,
            "resource_type": "text",
            "description": f"Text content of arXiv paper {paper_id_str} (first {max_pages} pages)",
            "usage": "Reference this URI to get the paper's text content"
        }
    
    # Add summary
    result["summary"] = f"Generated {len(resource_types)} resource type(s) for paper {paper_id_str}"
    result["usage_instructions"] = [
        "Copy any URI from the results and reference it directly in your conversation",
        "Example: 'Please analyze this image: arxiv://1706.03762/image/1'",
        "Compatible MCP clients will automatically load the referenced resources"
    ]
    
    return result


# Custom HTTP routes
@mcp.custom_route("/arxiv/health", methods=["GET"])
async def arxiv_health_check(request):
    """
    Health check endpoint for arXiv server.

    Args:
        request: HTTP request object

    Returns:
        JSON response with server health status
    """
    from starlette.responses import JSONResponse

    # Test arXiv API connectivity
    try:
        test_response = await arxiv_client.get(
            ARXIV_API_BASE, params={"search_query": "test", "max_results": 1}
        )
        status = "healthy" if test_response.status_code == 200 else "degraded"
    except Exception:
        status = "unhealthy"

    return JSONResponse(
        {
            "status": status,
            "server": "arXiv MCP Server",
            "tools_count": len(await mcp.get_tools()),
            "resources_count": len(await mcp.get_resources()),
            "timestamp": datetime.now().isoformat(),
        }
    )


def create_arxiv_server():
    """Create and configure the arXiv MCP server."""
    return mcp


def run_server(
    transport: Literal["stdio", "http", "sse", "streamable-http"] = "stdio",
    show_banner=True,
    host="0.0.0.0",
    port=8000,
):
    """Run the arXiv server with custom configuration."""
    server = create_arxiv_server()

    print("Starting arXiv MCP Server")
    print(f"Transport: {transport}")

    if transport in ["http", "sse", "streamable-http"]:
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"URL: http://{host}:{port}")
        print("Available endpoints:")
        print("  - /mcp - MCP server")
        print("  - /health - Health check")
        print("-" * 50)

    # Run the server with custom settings
    if transport in ["http", "sse", "streamable-http"]:
        server.run(transport=transport, show_banner=show_banner, host=host, port=port)
    else:
        server.run(transport=transport, show_banner=show_banner)


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="arXiv MCP Server - Access arXiv papers with search, download, and reading capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run with stdio transport (for MCP clients like Claude Desktop)
            python arxiv_server.py

            # Run with HTTP transport on localhost:8000
            python arxiv_server.py --transport http

            # Run on all interfaces, port 9000
            python arxiv_server.py --transport http --host 0.0.0.0 --port 9000
        """,
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol to use (default: stdio)",
    )

    parser.add_argument(
        "--no-banner", action="store_true", help="Disable server banner display"
    )

    args = parser.parse_args()

    try:
        run_server(
            host=args.host,
            port=args.port,
            transport=args.transport,
            show_banner=not args.no_banner,
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
