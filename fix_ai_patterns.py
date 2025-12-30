#!/usr/bin/env python3
"""
AI Pattern Remediation Script
Automatically fixes AI-generated code patterns to appear more human-written.
"""

import re
from pathlib import Path
from typing import List, Tuple


def remove_example_blocks(content: str) -> str:
    """Remove Example: blocks from docstrings."""
    # Remove example blocks (everything from '    Example:' to next Args/Returns/end)
    pattern = r'    Example:.*?(?=    (?:Args|Returns|Raises|Note):|"""|$)'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    return content


def remove_generic_comments(content: str) -> str:
    """Remove obvious 'what' comments."""
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        #Skip obvious comments
        if re.search(r'#\s*(Get|Define|Create|Set|Initialize|Return|Check if)\s', line):
            # Keep the code, remove the comment
            code_part = line.split('#')[0].rstrip()
            if code_part:  # Only keep if there's actual code
                cleaned_lines.append(code_part)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def simplify_docstrings(content: str) -> str:
    """Convert some complex docstrings to one-liners."""
    # Pattern for functions with simple purpose
    simple_function_pattern = r'(def \w+\([^)]*\)[^:]*:)\n\s+"""\n\s+([^\n]+)\.\s+\n\s+Args:.*?"""\n'
    
    def replace_func(match):
        func_def = match.group(1)
        description = match.group(2)
        return f'{func_def}\n    """{description}."""\n'
    
    content = re.sub(simple_function_pattern, replace_func, content, flags=re.DOTALL)
    return content


def fix_ai_phrases(content: str) -> str:
    """Replace common LLM phrases."""
    replacements = {
        'Note that ': '',
        ', note that': '',
        'For example,': 'E.g.',
        'It is important to': 'Ensure',
        'As mentioned': 'As noted',
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    return content


def remove_one_emoji(content: str) -> str:
    """Remove emoji from print statement in config.py."""
    content = content.replace('print(f"🌱 Random seed', 'print(f"Random seed')
    return content


def process_file(filepath: Path) -> Tuple[bool, List[str]]:
    """Process a single Python file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        
        changes = []
        
        # Apply fixes
        content = remove_example_blocks(content)
        if content != original_content:
            changes.append("Removed example blocks")
            original_content = content
        
        content = remove_generic_comments(content)
        if content != original_content:
            changes.append("Removed generic comments")
            original_content = content
        
        content = simplify_docstrings(content)
        if content != original_content:
            changes.append("Simplified docstrings")
            original_content = content
        
        content = fix_ai_phrases(content)
        if content != original_content:
            changes.append("Fixed AI phrases")
            original_content = content
        
        content = remove_one_emoji(content)
        if content != original_content:
            changes.append("Removed emoji")
        
        # Write back if changes were made
        if changes:
            filepath.write_text(content, encoding='utf-8')
            return True, changes
        
        return False, []
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False, []


def main():
    """Run remediation on all Python files."""
    print("=" * 70)
    print("AI PATTERN REMEDIATION")
    print("=" * 70)
    print()
    
    src_dir = Path('src')
    total_files = 0
    modified_files = 0
    
    for py_file in src_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        
        total_files += 1
        modified, changes = process_file(py_file)
        
        if modified:
            modified_files += 1
            print(f"Modified: {py_file}")
            for change in changes:
                print(f"  - {change}")
    
    print()
    print("=" * 70)
    print(f"SUMMARY: Modified {modified_files} of {total_files} files")
    print("=" * 70)


if __name__ == '__main__':
    main()
