import React from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "@fortawesome/fontawesome-free/css/all.min.css";

const JSONExporterBackEndSupported = ({
  fileName,
  addModeCoordinates,
  deleteModeCoordinates,
}) => {
  const exportJSON = async () => {
    try {
      const response = await axios.post("http://localhost:3001/export", {
        fileName,
        data: { addModeCoordinates, deleteModeCoordinates },
      });
      toast.success(
        `${fileName}_edited.json has been exported successfully to the 'export' folder`,
        {
          className: "custom-toast",
        }
      );
    } catch (err) {
      toast.error(`Error exporting file: ${fileName}_edited.json`, {
        className: "custom-toast",
      });
    }
  };

  return (
    <button className="export-button" onClick={exportJSON}>
      <i className="fa-solid fa-file-export fa-md"></i>Export
    </button>
  );
};

export default JSONExporterBackEndSupported;
