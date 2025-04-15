import cv2
from image_searcher import ImageSearcher

def main():
    # Initialize the searcher
    searcher = ImageSearcher(
        index_path="models/siglip2b-16-256-rotated.index", 
        meta_path="models/siglip2b-16-256-rotated-metadata.pkl"
    )
    
    # Load an example frame with OpenCV
    frame = cv2.imread("images/choco_knots_cart.jpg")
    
    if frame is not None:
        # Perform the search using the frame
        results = searcher.search(frame, top_k=5)
        
        print("\nTop Matches:")
        for i, (path, score) in enumerate(results, 1):
            print(f"{i}. {path} (Score: {score})")
    else:
        print("Error loading image")

if __name__ == "__main__":
    main()