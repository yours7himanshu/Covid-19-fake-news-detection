import os
import pandas as pd
import glob

dataset_dir = 'datasets'

def analyze_datasets():
    files = glob.glob(os.path.join(dataset_dir, '*.csv'))
    
    summary = {
        'total_files': len(files),
        'total_rows': 0,
        'breakdown': {},
        'columns': {}
    }
    
    print(f"Found {len(files)} CSV files.")
    print("-" * 50)

    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            # Read only header first to check columns, then count rows
            # Using on_bad_lines='skip' to avoid crashing on malformed rows if any
            df = pd.read_csv(file_path, on_bad_lines='skip')
            
            row_count = len(df)
            cols = list(df.columns)
            
            summary['total_rows'] += row_count
            summary['breakdown'][filename] = row_count
            
            # Group by file type pattern (e.g., ClaimFake, NewsReal)
            # Simple heuristic: split by '_' and take first part or first few parts
            file_type = filename.split('_')[0] # e.g. ClaimFakeCOVID-19
            if 'tweets' in filename:
                if 'replies' in filename:
                    file_type += '_tweets_replies'
                else:
                    file_type += '_tweets'
            
            if file_type not in summary['columns']:
                summary['columns'][file_type] = cols
                print(f"\nSchema for {file_type} ({filename}):")
                print(cols)
                if not df.empty:
                    print("Sample row:")
                    print(df.iloc[0].to_dict())

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print("-" * 50)
    print("Summary Statistics:")
    print(f"Total Rows across all files: {summary['total_rows']}")
    
    # Aggregate counts
    categories = {
        'ClaimFake': 0,
        'ClaimReal': 0,
        'NewsFake': 0,
        'NewsReal': 0
    }
    
    subtypes = {
        'Content': 0,
        'Tweets': 0,
        'Replies': 0
    }

    for filename, count in summary['breakdown'].items():
        print(f"{filename}: {count}")
        
        for cat in categories:
            if filename.startswith(cat):
                categories[cat] += count
        
        if 'tweets' in filename:
            if 'replies' in filename:
                subtypes['Replies'] += count
            else:
                subtypes['Tweets'] += count
        else:
            subtypes['Content'] += count

    print("\nCategory Breakdown:")
    for cat, count in categories.items():
        print(f"{cat}: {count}")

    print("\nData Type Breakdown:")
    for type_, count in subtypes.items():
        print(f"{type_}: {count}")

if __name__ == "__main__":
    analyze_datasets()
