import { useEffect } from "react";

const useKeyboardShortcuts = ({
    toggleAddMode,
    toggleDeleteMode,
    toggleMarkUnusableMode,
    undoLastCoordinate
}) => {
      const handleKeyDown = (event) => {
        if (event.key === "A" || event.key === "a") {
          toggleAddMode();
        } else if (event.key === "D" || event.key === "d") {
          toggleDeleteMode();
        } else if (event.key === "U" || event.key === "u") {
          toggleMarkUnusableMode();
        } else if (event.ctrlKey && event.key === "z") {
          undoLastCoordinate();
        }
      };
    
      useEffect(() => {
        // Checks for key presses
        window.addEventListener("keydown", handleKeyDown);
    
        return () => {
          window.removeEventListener("keydown", handleKeyDown);
        };
      });
}

export default useKeyboardShortcuts;