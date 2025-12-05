import os

def get_ecoinvent_credentials():
    """Get ecoinvent credentials from environment variables or manual specification."""
    # Try environment variables first
    username = os.getenv("ECOINVENT_USERNAME")
    password = os.getenv("ECOINVENT_PASSWORD")
    
    # Fallback to manual specification (uncomment lines above if needed)
    if not username or not password:
        try:
            return globals()['username'], globals()['password']
        except KeyError:
            print("⚠️  Please uncomment and set username/password variables above")
            return None, None
    
    return username, password