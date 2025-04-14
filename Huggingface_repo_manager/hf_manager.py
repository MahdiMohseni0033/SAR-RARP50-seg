#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██╗  ██╗███████╗    ███╗   ███╗ █████╗ ███╗   ██╗ █████╗  ██████╗ ███████╗██████╗  ║
║   ██║  ██║██╔════╝    ████╗ ████║██╔══██╗████╗  ██║██╔══██╗██╔════╝ ██╔════╝██╔══██╗ ║
║   ███████║█████╗      ██╔████╔██║███████║██╔██╗ ██║███████║██║  ███╗█████╗  ██████╔╝ ║
║   ██╔══██║██╔══╝      ██║╚██╔╝██║██╔══██║██║╚██╗██║██╔══██║██║   ██║██╔══╝  ██╔══██╗ ║
║   ██║  ██║██║         ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║  ██║╚██████╔╝███████╗██║  ██║ ║
║   ╚═╝  ╚═╝╚═╝         ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝ ║
║                                                                           ║
║   Huggingface Repository Manager - A CLI tool to manage HF repositories   ║
║   Version: 1.0.0                                                          ║
║   Author: Your Organization                                               ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

# Standard library imports
import os
import sys
import getpass
import pprint
from typing import Optional, Dict, List, Any, Union

# Third-party imports
from huggingface_hub import HfApi
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    HFValidationError,
    logging as hf_logging,
    HfHubHTTPError,
)

# ┌─────────────────────────────────────────────────────────────────┐
# │ Configuration                                                    │
# └─────────────────────────────────────────────────────────────────┘
# Set logging level (optional, adjust as needed: DEBUG, INFO, WARNING, ERROR)
# Use WARNING for less verbose output from the huggingface_hub library
hf_logging.set_verbosity_warning()


# ANSI color codes for terminal formatting
class Colors:
    """ANSI color codes for prettier terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ┌─────────────────────────────────────────────────────────────────┐
# │ Core API Functions                                               │
# └─────────────────────────────────────────────────────────────────┘
def initialize_api(token: Optional[str] = None) -> HfApi:
    """
    Initializes the Hugging Face API client.

    Args:
        token: The Hugging Face API token. If None, will attempt to use
               environment variables or login cache.

    Returns:
        HfApi: An authenticated Hugging Face API client.

    Raises:
        SystemExit: If authentication fails.
    """
    print(f"{Colors.HEADER}Initializing Hugging Face API...{Colors.ENDC}")
    try:
        api = HfApi(token=token)
        # Attempt a quick check to see if the token seems valid and get user info
        user = api.whoami()
        print(f"{Colors.GREEN}✓ Authenticated as:{Colors.ENDC} {Colors.BOLD}{user.get('name', 'N/A')}{Colors.ENDC} "
              f"({user.get('email', 'email not public')})")
        return api
    except HFValidationError as e:
        print(f"\n{Colors.FAIL}Error: Invalid Hugging Face token or configuration.{Colors.ENDC} {e}", file=sys.stderr)
        print(
            f"{Colors.WARNING}Please ensure your token is correct, or you are logged in (`huggingface-cli login`).{Colors.ENDC}",
            file=sys.stderr)
        sys.exit(1)
    except HfHubHTTPError as e:
        # Often indicates network issues or authentication problems
        print(f"\n{Colors.FAIL}Error: Could not connect or authenticate with Hugging Face Hub.{Colors.ENDC} {e}",
              file=sys.stderr)
        print(f"{Colors.WARNING}Check your network connection and token permissions.{Colors.ENDC}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}Error initializing HfApi: {type(e).__name__}:{Colors.ENDC} {e}", file=sys.stderr)
        print(f"{Colors.WARNING}Ensure `huggingface-hub` library is installed correctly.{Colors.ENDC}", file=sys.stderr)
        sys.exit(1)


# ┌─────────────────────────────────────────────────────────────────┐
# │ Repository Operations                                            │
# └─────────────────────────────────────────────────────────────────┘
def upload_file(api: HfApi, repo_id: str, repo_type: str) -> None:
    """
    Handles the file upload process.

    Args:
        api: Initialized HfApi client
        repo_id: Target repository ID (e.g., 'username/repo-name')
        repo_type: Repository type ('model', 'dataset', or 'space')
    """
    print(f"\n{Colors.HEADER}╔═══════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.HEADER}║       📤 UPLOAD FILE          ║{Colors.ENDC}")
    print(f"{Colors.HEADER}╚═══════════════════════════════╝{Colors.ENDC}")

    local_path = input(f"{Colors.BOLD}Enter the full path to the local file to upload:{Colors.ENDC} ")
    local_path = local_path.strip()  # Remove leading/trailing whitespace

    if not os.path.isfile(local_path):
        print(f"\n{Colors.FAIL}Error: Local file not found at '{os.path.abspath(local_path)}'{Colors.ENDC}",
              file=sys.stderr)
        return  # Go back to menu

    default_path_in_repo = os.path.basename(local_path)
    path_in_repo = input(
        f"{Colors.BOLD}Enter the desired path within the repo (e.g., 'models/v1/weights.pt'){Colors.ENDC}\n"
        f"[Leave blank to use '{default_path_in_repo}']: ")
    path_in_repo = path_in_repo.strip() or default_path_in_repo

    default_commit_message = f"Upload {os.path.basename(local_path)}"
    commit_message = input(
        f"{Colors.BOLD}Enter commit message{Colors.ENDC} [Leave blank for '{default_commit_message}']: ")
    commit_message = commit_message.strip() or default_commit_message

    print(f"\n{Colors.CYAN}Preparing to upload:{Colors.ENDC}")
    print(f"  • Local file:  {Colors.BOLD}{local_path}{Colors.ENDC}")
    print(f"  • Destination: {Colors.BOLD}{repo_id}/{path_in_repo}{Colors.ENDC} ({repo_type})")
    print(f"  • Message:     {commit_message}")

    try:
        upload_url = api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        print(f"\n{Colors.GREEN}✅ Successfully uploaded!{Colors.ENDC}")
        print(f"   {Colors.UNDERLINE}File URL:{Colors.ENDC} {upload_url}")
    except RepositoryNotFoundError:
        print(f"\n{Colors.FAIL}Error: Repository '{repo_id}' not found or you lack write access.{Colors.ENDC}",
              file=sys.stderr)
    except FileNotFoundError:  # Should be caught above, but just in case
        print(f"\n{Colors.FAIL}Error: Local file not found at '{os.path.abspath(local_path)}'{Colors.ENDC}",
              file=sys.stderr)
    except HFValidationError as e:
        print(f"\n{Colors.FAIL}Error: Invalid input for upload.{Colors.ENDC} {e}", file=sys.stderr)
    except HfHubHTTPError as e:
        print(f"\n{Colors.FAIL}Error: Upload failed due to HTTP error.{Colors.ENDC} {e}", file=sys.stderr)
        print(f"{Colors.WARNING}Check your connection, token permissions, and if the repository exists.{Colors.ENDC}",
              file=sys.stderr)
    except Exception as e:
        print(f"\n{Colors.FAIL}Upload failed: An unexpected error occurred: {type(e).__name__}:{Colors.ENDC} {e}",
              file=sys.stderr)


def delete_file(api: HfApi, repo_id: str, repo_type: str) -> None:
    """
    Handles the file deletion process.

    Args:
        api: Initialized HfApi client
        repo_id: Target repository ID (e.g., 'username/repo-name')
        repo_type: Repository type ('model', 'dataset', or 'space')
    """
    print(f"\n{Colors.HEADER}╔═══════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.HEADER}║       🗑️  DELETE FILE         ║{Colors.ENDC}")
    print(f"{Colors.HEADER}╚═══════════════════════════════╝{Colors.ENDC}")

    path_in_repo = input(
        f"{Colors.BOLD}Enter the exact path/filename within the repo to delete{Colors.ENDC} (e.g., 'checkpoints/old.pt'): ")
    path_in_repo = path_in_repo.strip()

    if not path_in_repo:
        print(f"\n{Colors.FAIL}Error: Path in repository cannot be empty.{Colors.ENDC}", file=sys.stderr)
        return  # Go back to menu

    default_commit_message = f"Delete {path_in_repo}"
    commit_message = input(
        f"{Colors.BOLD}Enter commit message{Colors.ENDC} [Leave blank for '{default_commit_message}']: ")
    commit_message = commit_message.strip() or default_commit_message

    print(f"{Colors.WARNING}{'-' * 60}{Colors.ENDC}")
    print(f"{Colors.WARNING}Repository:      {repo_id} ({repo_type}){Colors.ENDC}")
    print(f"{Colors.WARNING}File to delete:  {path_in_repo}{Colors.ENDC}")
    print(f"{Colors.FAIL}!!! WARNING: This operation is IRREVERSIBLE !!!{Colors.ENDC}")
    print(f"{Colors.WARNING}{'-' * 60}{Colors.ENDC}")

    confirm = input(f"{Colors.BOLD}Proceed with deletion? (yes/no):{Colors.ENDC} ").strip().lower()
    if confirm != 'yes':
        print(f"{Colors.CYAN}Deletion cancelled by user.{Colors.ENDC}")
        return

    print(f"\n{Colors.CYAN}Attempting to delete '{path_in_repo}' from '{repo_id}'...{Colors.ENDC}")
    try:
        api.delete_file(
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        print(f"\n{Colors.GREEN}✅ Successfully deleted '{path_in_repo}' from '{repo_id}'.{Colors.ENDC}")
    except EntryNotFoundError:
        print(f"\n{Colors.FAIL}Error: File '{path_in_repo}' not found in repository '{repo_id}'.{Colors.ENDC}",
              file=sys.stderr)
    except RepositoryNotFoundError:
        print(f"\n{Colors.FAIL}Error: Repository '{repo_id}' not found or you lack write access.{Colors.ENDC}",
              file=sys.stderr)
    except HFValidationError as e:
        print(f"\n{Colors.FAIL}Error: Invalid input for deletion.{Colors.ENDC} {e}", file=sys.stderr)
    except HfHubHTTPError as e:
        print(f"\n{Colors.FAIL}Error: Deletion failed due to HTTP error.{Colors.ENDC} {e}", file=sys.stderr)
        print(
            f"{Colors.WARNING}Check your connection, token permissions, and if the file/repository exists.{Colors.ENDC}",
            file=sys.stderr)
    except Exception as e:
        print(f"\n{Colors.FAIL}Deletion failed: An unexpected error occurred: {type(e).__name__}:{Colors.ENDC} {e}",
              file=sys.stderr)


def get_repo_info(api: HfApi, repo_id: str, repo_type: str) -> None:
    """
    Fetches and displays repository information.

    Args:
        api: Initialized HfApi client
        repo_id: Target repository ID (e.g., 'username/repo-name')
        repo_type: Repository type ('model', 'dataset', or 'space')
    """
    print(f"\n{Colors.HEADER}╔═══════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.HEADER}║    📊 REPOSITORY INFO        ║{Colors.ENDC}")
    print(f"{Colors.HEADER}╚═══════════════════════════════╝{Colors.ENDC}")

    print(f"{Colors.CYAN}Fetching info for '{repo_id}' ({repo_type})...{Colors.ENDC}")
    try:
        # NOTE: repo_info returns ModelInfo, DatasetInfo or SpaceInfo depending on repo_type
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type)

        print(f"\n{Colors.BOLD}Repository Details:{Colors.ENDC}")
        # Use the repo_type variable passed to the function, not info.repo_type
        # Use getattr or .get for potentially missing attributes for robustness
        details = {
            "ID": getattr(info, 'id', 'N/A'),
            "Type": repo_type,  # Use the function's argument
            "Private": getattr(info, 'private', 'N/A'),
            "Author": getattr(info, 'author', 'N/A'),
            "Last Modified": info.last_modified.strftime('%Y-%m-%d %H:%M:%S %Z') if getattr(info, 'last_modified',
                                                                                            None) else "N/A",
            "SHA": getattr(info, 'sha', 'N/A'),
            "Downloads (last month)": getattr(info, 'downloads', 'N/A'),
            "Likes": getattr(info, 'likes', 'N/A'),
            "Tags": getattr(info, 'tags', []),
            "Card/README": bool(getattr(info, 'card_data', None) or getattr(info, 'readme', None)),
            "Siblings (Files/Folders) Count": len(getattr(info, 'siblings', []))  # Use getattr for siblings too
        }

        # Pretty print with colors
        for key, value in details.items():
            print(f"  {Colors.CYAN}{key}:{Colors.ENDC} {value}")

        # Manually construct the URL string based on repo_type and repo_id
        base_url = "https://huggingface.co"
        if repo_type == "space":
            repo_url_str = f"{base_url}/spaces/{repo_id}"
        # Add elif for dataset if its URL structure differs in the future
        # elif repo_type == "dataset":
        #    repo_url_str = f"{base_url}/{repo_id}" # Currently same as model
        else:  # Covers 'model' and 'dataset' for now
            repo_url_str = f"{base_url}/{repo_id}"

        print(f"\n{Colors.UNDERLINE}URL:{Colors.ENDC} {repo_url_str}")

    except RepositoryNotFoundError:
        print(f"\n{Colors.FAIL}Error: Repository '{repo_id}' not found or you lack read access.{Colors.ENDC}",
              file=sys.stderr)
    except HfHubHTTPError as e:
        print(f"\n{Colors.FAIL}Error: Failed to fetch repo info due to HTTP error.{Colors.ENDC} {e}", file=sys.stderr)
    except Exception as e:
        # Print the exception type along with the message for better debugging
        print(
            f"\n{Colors.FAIL}Failed to get repository info: An unexpected error occurred: {type(e).__name__}:{Colors.ENDC} {e}",
            file=sys.stderr)


def list_repo_tree(api: HfApi, repo_id: str, repo_type: str) -> None:
    """
    Fetches and displays the repository file tree.

    Args:
        api: Initialized HfApi client
        repo_id: Target repository ID (e.g., 'username/repo-name')
        repo_type: Repository type ('model', 'dataset', or 'space')
    """
    print(f"\n{Colors.HEADER}╔═══════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.HEADER}║      🔍 FILE EXPLORER         ║{Colors.ENDC}")
    print(f"{Colors.HEADER}╚═══════════════════════════════╝{Colors.ENDC}")

    print(f"{Colors.CYAN}Fetching file list for '{repo_id}' ({repo_type})...{Colors.ENDC}")
    try:
        # Use iterator=True for potentially very large repos to save memory,
        # though it requires converting to list for sorting/counting.
        # files = list(api.list_repo_files(repo_id=repo_id, repo_type=repo_type, iterator=True))
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

        if not files:
            print(
                f"\n{Colors.WARNING}Repository appears to be empty or contains only hidden files (like .gitattributes).{Colors.ENDC}")
            return

        print(f"\n{Colors.BOLD}Found {len(files)} file(s) in repository:{Colors.ENDC}")

        # Sort for better readability, especially with nested structures
        files.sort()

        # Create a more visual tree-like structure
        current_dir = ""
        for filename in files:
            # Check if we're in a new directory
            dir_parts = os.path.dirname(filename).split("/")

            if dir_parts[0] != current_dir and dir_parts[0] != "":
                # We're in a new top-level directory
                current_dir = dir_parts[0]
                print(f"\n{Colors.BLUE}📁 {current_dir}/{Colors.ENDC}")

            # Determine file type for icon (very simple version)
            if filename.endswith(('.py', '.js', '.cpp', '.c', '.java', '.sh')):
                icon = "📜"  # Script file
            elif filename.endswith(('.md', '.txt', '.rst')):
                icon = "📄"  # Document
            elif filename.endswith(('.jpg', '.png', '.gif', '.bmp')):
                icon = "🖼️"  # Image
            elif filename.endswith(('.pt', '.pth', '.bin', '.onnx', '.pb')):
                icon = "🧠"  # Model file
            elif filename.endswith(('.json', '.yaml', '.yml', '.toml')):
                icon = "⚙️"  # Config file
            elif filename.endswith(('.csv', '.tsv', '.xlsx', '.parquet')):
                icon = "📊"  # Data file
            elif os.path.basename(filename) == ".gitattributes":
                icon = "🔧"  # Git file
            else:
                icon = "📋"  # Generic file

            # Print with appropriate indentation based on directory depth
            indent = "   " * (len(dir_parts) - 1 if dir_parts[0] != "" else 0)
            if dir_parts[0] == current_dir or dir_parts[0] == "":
                print(f"  {indent}{icon} {os.path.basename(filename)}")

    except RepositoryNotFoundError:
        print(f"\n{Colors.FAIL}Error: Repository '{repo_id}' not found or you lack read access.{Colors.ENDC}",
              file=sys.stderr)
    except HfHubHTTPError as e:
        print(f"\n{Colors.FAIL}Error: Failed to list files due to HTTP error.{Colors.ENDC} {e}", file=sys.stderr)
    except Exception as e:
        print(
            f"\n{Colors.FAIL}Failed to list repository files: An unexpected error occurred: {type(e).__name__}:{Colors.ENDC} {e}",
            file=sys.stderr)


# ┌─────────────────────────────────────────────────────────────────┐
# │ Main Application Logic                                           │
# └─────────────────────────────────────────────────────────────────┘
def display_menu(repo_id: str, repo_type: str) -> None:
    """
    Prints the main menu options.

    Args:
        repo_id: Current repository ID
        repo_type: Current repository type
    """
    print(f"\n{Colors.HEADER}{'═' * 60}{Colors.ENDC}")
    print(
        f"{Colors.HEADER}  🤗 Hugging Face Manager | Repo: {Colors.BOLD}{repo_id}{Colors.ENDC} {Colors.HEADER}({repo_type}){Colors.ENDC}")
    print(f"{Colors.HEADER}{'═' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}Please choose an action:{Colors.ENDC}")
    print(f"  {Colors.CYAN}1.{Colors.ENDC} 📤 Upload a file")
    print(f"  {Colors.CYAN}2.{Colors.ENDC} 🗑️  Delete a file")
    print(f"  {Colors.CYAN}3.{Colors.ENDC} 📊 Show repository information")
    print(f"  {Colors.CYAN}4.{Colors.ENDC} 🔍 List repository files (tree view)")
    print(f"  {Colors.CYAN}5.{Colors.ENDC} 🚪 Exit")
    print(f"{Colors.HEADER}{'-' * 60}{Colors.ENDC}")


def main() -> None:
    """Main function to handle user interaction."""
    # Display welcome banner
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}          Welcome to the Hugging Face Repository Manager!{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}")

    # --- Get Initial Configuration ---
    repo_id = input(f"\n{Colors.BOLD}Enter the Repository ID{Colors.ENDC} (e.g., 'username/my-cool-model'): ").strip()
    while not repo_id:
        print(f"{Colors.WARNING}Repository ID cannot be empty.{Colors.ENDC}")
        repo_id = input(f"{Colors.BOLD}Enter the Repository ID:{Colors.ENDC} ").strip()

    print(f"\n{Colors.BOLD}Enter your Hugging Face API token.{Colors.ENDC}")
    print(
        f"{Colors.CYAN}You can get one from https://huggingface.co/settings/tokens (needs 'write' permissions for upload/delete){Colors.ENDC}")
    print(
        f"{Colors.CYAN}Leave blank to try using environment variable (HF_TOKEN) or login cache (`huggingface-cli login`).{Colors.ENDC}")
    token = getpass.getpass(f"{Colors.BOLD}Token:{Colors.ENDC} ").strip()
    token = token if token else None  # Use None if blank, so HfApi uses default methods

    repo_type = ""
    valid_repo_types = ["model", "dataset", "space"]
    while repo_type not in valid_repo_types:
        repo_type_input = input(
            f"{Colors.BOLD}Enter repository type{Colors.ENDC} ({'/'.join(valid_repo_types)}) [default: model]: ").strip().lower()
        if not repo_type_input:
            repo_type = "model"  # Default value
        elif repo_type_input in valid_repo_types:
            repo_type = repo_type_input
        else:
            print(f"{Colors.WARNING}Invalid type. Please choose from: {', '.join(valid_repo_types)}.{Colors.ENDC}")

    # --- Initialize API ---
    api = initialize_api(token=token)  # Exits if initialization fails
    print(f"\n{Colors.GREEN}Successfully configured for repository: '{repo_id}' ({repo_type}){Colors.ENDC}")

    # --- Main Action Loop ---
    while True:
        display_menu(repo_id, repo_type)  # Show repo in menu title
        choice = input(f"{Colors.BOLD}Enter your choice (1-5):{Colors.ENDC} ").strip()

        if choice == '1':
            upload_file(api, repo_id, repo_type)
        elif choice == '2':
            delete_file(api, repo_id, repo_type)
        elif choice == '3':
            get_repo_info(api, repo_id, repo_type)
        elif choice == '4':
            list_repo_tree(api, repo_id, repo_type)
        elif choice == '5':
            print(f"\n{Colors.GREEN}Exiting program. Goodbye!{Colors.ENDC}")
            break
        else:
            print(f"{Colors.WARNING}Invalid choice. Please enter a number between 1 and 5.{Colors.ENDC}")

        if choice != '5':
            input(f"\n{Colors.CYAN}Press Enter to return to the menu...{Colors.ENDC}")


if __name__ == "__main__":
    main()