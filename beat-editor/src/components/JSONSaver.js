import React from "react";
import axios from "axios";
import { toast } from "react-toastify";
import "@fortawesome/fontawesome-free/css/all.min.css";

const JSONSaver = ({ fileName, addModeCoordinates, deleteModeCoordinates, unusableSegments }) => {
  const saveJSON = async () => {
    try {
      const response = await axios.post("http://localhost:3001/saved", {
        fileName,
        data: { addModeCoordinates, deleteModeCoordinates, unusableSegments },
      });
      if (
        addModeCoordinates.length !== 0 &&
        deleteModeCoordinates.length !== 0 &&
        unusableSegments.length !== 0
      ) {
        toast.success(`${fileName}_edited.json has been saved`, {
          className: "custom-toast",
        });
      }
    } catch (err) {
      toast.error(`Error saving file: ${fileName}_edited.json`, {
        className: "custom-toast",
      });
    }
  };

  return (
    <button className="save-button" onClick={saveJSON}>
      <i className="fa-solid fa-save fa-md"></i>Save
    </button>
  );
};

export default JSONSaver;
