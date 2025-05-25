"""
Fix timezone synchronization across all files
Standardize timestamp handling to get dashboard working
"""

import os
import re

def sync_timezone_handling():
    """Synchronize all timezone handling to remove UTC conflicts"""
    
    print("🔧 Synchronizing timezone handling...")
    
    # Fix app.py timestamp handling
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Remove all UTC timezone specifications
    content = re.sub(r"pd\.to_datetime\([^)]+, utc=True\)", 
                     lambda m: m.group(0).replace(", utc=True", ""), content)
    
    # Ensure consistent datetime parsing
    content = content.replace(
        "df['datetime'] = pd.to_datetime(df['Local time'], utc=True)",
        "df['datetime'] = pd.to_datetime(df['Local time'])"
    )
    
    with open('app.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed app.py timezone handling")
    
    # Fix strategy_tester.py if it has timezone issues
    if os.path.exists('strategy_tester.py'):
        with open('strategy_tester.py', 'r') as f:
            strategy_content = f.read()
        
        # Remove UTC specifications
        strategy_content = re.sub(r"pd\.to_datetime\([^)]+, utc=True\)", 
                                lambda m: m.group(0).replace(", utc=True", ""), strategy_content)
        
        with open('strategy_tester.py', 'w') as f:
            f.write(strategy_content)
        
        print("✅ Fixed strategy_tester.py timezone handling")
    
    print("🎯 All timezone handling synchronized!")
    print("Your dashboard should now display clean zero baselines")

if __name__ == "__main__":
    sync_timezone_handling()