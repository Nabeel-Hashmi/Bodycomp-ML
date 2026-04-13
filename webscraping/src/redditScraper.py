import requests
import os
import time

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SUBREDDIT = "guessmybf"

def download_all_images(subreddit, max_pages=10, output_dir="../data/raw/reddit_images"):
    os.makedirs(output_dir, exist_ok=True)
    after = None
    total = 0

    for page in range(max_pages):
        url = f"https://www.reddit.com/r/{subreddit}.json?limit=25"
        if after:
            url += f"&after={after}"

        response = requests.get(url, headers=HEADERS)
        data = response.json()
        posts = data["data"]["children"]

        for post in posts:
            d = post["data"]
            post_id = d["id"]

            # Skip non-gallery, non-image posts
            if not d.get("is_gallery") and d.get("post_hint") != "image":
                continue

            # --- Gallery posts (your example) ---
            if d.get("is_gallery") and "media_metadata" in d:
                # Use gallery_data to get correct image order
                items = d.get("gallery_data", {}).get("items", [])
                for i, item in enumerate(items):
                    media_id = item["media_id"]
                    meta = d["media_metadata"].get(media_id, {})

                    if meta.get("status") != "valid":
                        continue

                    # Get highest res URL and clean the &amp;
                    img_url = meta["s"]["u"].replace("&amp;", "&")
                    ext = ".jpg"
                    filename = f"{output_dir}/{post_id}_{i+1}{ext}"

                    try:
                        img_data = requests.get(img_url, headers=HEADERS).content
                        with open(filename, "wb") as f:
                            f.write(img_data)
                        total += 1
                        print(f"✓ {filename}")
                    except Exception as e:
                        print(f"✗ Failed {media_id}: {e}")

                    time.sleep(0.3)  # be polite

            # --- Single image posts ---
            elif d.get("post_hint") == "image":
                img_url = d["url"]
                filename = f"{output_dir}/{post_id}_1.jpg"
                try:
                    img_data = requests.get(img_url, headers=HEADERS).content
                    with open(filename, "wb") as f:
                        f.write(img_data)
                    total += 1
                    print(f"✓ {filename}")
                except Exception as e:
                    print(f"✗ Failed: {e}")

        after = data["data"].get("after")
        if not after:
            break

        print(f"--- Page {page+1} done, moving to next... ---")
        time.sleep(2)  # pause between pages

    print(f"\nDone! {total} images saved to '{output_dir}/'")

download_all_images("guessmybf")