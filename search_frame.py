import cv2
from image_searcher import ImageSearcher
import time

def main():
    start_time = time.time()
    # Initialize the searcher
    searcher = ImageSearcher(
        index_path="models/siglip2b-16-256-rotated.index", 
        meta_path="models/siglip2b-16-256-rotated-metadata.pkl",
        model_name="google/siglip2-base-patch16-256"
    )
    
    # Load an example frame with OpenCV
    
    frame = cv2.imread("images/555 Tuna_Caldereta.png")
    frame = cv2.resize(frame, (256, 256))  # resize depending on the model
    if frame is not None:
        # Perform the search using the frame
        results = searcher.search(frame, top_k=5)
        
        print("\nTop Matches:")
        for i, (path, score) in enumerate(results, 1):
            print(f"{i}. {path} (Score: {score})")
    else:
        print("Error loading image")

    print(f"Search completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()