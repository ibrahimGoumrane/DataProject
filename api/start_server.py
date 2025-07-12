# -*- coding: utf-8 -*-
"""
Startup script for MCP server with proper Windows encoding handling
"""
import sys
import os
import locale

def setup_encoding():
    """Setup proper UTF-8 encoding for Windows"""
    if sys.platform.startswith('win'):
        # Set environment variables for UTF-8
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '1'
        
        # Try to set UTF-8 console encoding
        try:
            # For Windows 10 and newer
            os.system('chcp 65001 >nul 2>&1')
        except:
            pass
        
        # Set locale to UTF-8 if possible
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except:
            try:
                locale.setlocale(locale.LC_ALL, 'C.UTF-8')
            except:
                pass

def main():
    """Main entry point with encoding setup"""
    setup_encoding()
    
    # Import and run the server after encoding setup
    from server import mcp
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
