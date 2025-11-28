import pandas as pd
import glob
import os

def check_content_quality():
    files = glob.glob(os.path.join('datasets', 'NewsFake*.csv'))
    # Filter out tweets/replies
    content_files = [f for f in files if 'tweets' not in f]
    
    print(f"Checking content quality in: {content_files}")
    
    for f in content_files:
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
            if 'content' in df.columns:
                # Check for short content or specific markers
                total = len(df)
                valid_content = df['content'].dropna()
                short_content = valid_content[valid_content.str.len() < 100]
                login_content = valid_content[valid_content.str.contains('log in|sign up|subscribe', case=False, na=False)]
                
                print(f"\nFile: {os.path.basename(f)}")
                print(f"Total rows: {total}")
                print(f"Non-null content: {len(valid_content)}")
                print(f"Short content (<100 chars): {len(short_content)}")
                print(f"Potential login/paywall: {len(login_content)}")
                
                if not short_content.empty:
                    print("Sample short content:", short_content.iloc[0])
        except Exception as e:
            print(f"Error reading {f}: {e}")

if __name__ == "__main__":
    check_content_quality()
