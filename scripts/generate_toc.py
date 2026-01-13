#!/usr/bin/env python3
"""
Generate Table of Contents for index.html
Scans subdirectories and extracts titles from their index.html files
"""

import os
import re
from pathlib import Path
from urllib.parse import quote


# Directories to exclude from TOC generation
EXCLUDED_DIRS = {
    'assets',
    'DOCUMENTS',
    '.git',
    '.github',
    'scripts',
    'node_modules',
    '__pycache__',
}


def extract_title_from_html(html_path: Path) -> str | None:
    """Extract the <title> content from an HTML file."""
    try:
        content = html_path.read_text(encoding='utf-8')
        match = re.search(r'<title>([^<]+)</title>', content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"Warning: Could not read {html_path}: {e}")
    return None


def get_article_dirs(root_path: Path) -> list[tuple[str, str]]:
    """
    Get all article directories with their titles.
    Returns list of (folder_name, title) tuples sorted alphabetically.
    """
    articles = []
    
    for item in root_path.iterdir():
        # Skip non-directories and excluded directories
        if not item.is_dir():
            continue
        if item.name in EXCLUDED_DIRS:
            continue
        if item.name.startswith('.'):
            continue
            
        # Check for index.html
        index_html = item / 'index.html'
        if not index_html.exists():
            continue
            
        # Extract title
        title = extract_title_from_html(index_html)
        if title:
            articles.append((item.name, title))
    
    # Sort alphabetically by folder name (case-insensitive)
    articles.sort(key=lambda x: x[0].lower())
    
    return articles


def generate_toc_html(articles: list[tuple[str, str]]) -> str:
    """Generate the TOC list HTML."""
    items = []
    
    for idx, (folder_name, title) in enumerate(articles, start=1):
        # Use folder name as main title, HTML title as subtitle
        main_title = folder_name.replace('-', ' ')
        subtitle = title
        
        # URL encode the folder name to handle spaces and special characters
        encoded_folder = quote(folder_name, safe='')
        
        item_html = f'''            <li class="toc-item">
                <a href="./{encoded_folder}/" class="toc-link">
                    <span class="toc-number">{idx:02d}.</span>
                    <span class="toc-title-text">{main_title}</span>
                    <span class="toc-dots"></span>
                    <span class="toc-subtitle">{subtitle}</span>
                </a>
            </li>'''
        items.append(item_html)
    
    return '\n            \n'.join(items)


def update_index_html(root_path: Path, toc_html: str) -> bool:
    """Update the index.html file with the new TOC."""
    index_path = root_path / 'index.html'
    
    try:
        content = index_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {index_path}: {e}")
        return False
    
    # Pattern to match the TOC list content
    # Matches from <ul class="toc-list"> to </ul>
    pattern = r'(<ul class="toc-list">)\s*(.*?)\s*(</ul>)'
    
    replacement = f'\\1\n{toc_html}\n        \\3'
    
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
    
    if count == 0:
        print("Error: Could not find TOC list in index.html")
        return False
    
    try:
        index_path.write_text(new_content, encoding='utf-8')
        print(f"Successfully updated {index_path}")
        return True
    except Exception as e:
        print(f"Error writing {index_path}: {e}")
        return False


def main():
    # Get the repository root (parent of scripts directory)
    script_path = Path(__file__).resolve()
    root_path = script_path.parent.parent
    
    print(f"Scanning directory: {root_path}")
    
    # Get all article directories
    articles = get_article_dirs(root_path)
    
    if not articles:
        print("No articles found!")
        return 1
    
    print(f"Found {len(articles)} articles:")
    for folder, title in articles:
        print(f"  - {folder}: {title}")
    
    # Generate TOC HTML
    toc_html = generate_toc_html(articles)
    
    # Update index.html
    if update_index_html(root_path, toc_html):
        print("TOC generation complete!")
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())

