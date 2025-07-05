import React from 'react';
import { saveAs } from 'file-saver';
import '@fortawesome/fontawesome-free/css/all.min.css';

const JSONExporter = ({ fileName, addModeCoordinates, deleteModeCoordinates }) => {

    const exportJSON = () => {
        // Adding "Mode" key for addModeCoordinates and injecting the value with "ADD"
        const updatedAddModeCoordinates = addModeCoordinates.map(coordinate => ({
            ...coordinate,
            Mode: "ADD"
        }));

        // Adding "Mode" key for deleteModeCoordinates and injecting the value with "DELETE"
        const updatedDeleteModeCoordinates = deleteModeCoordinates.map(coordinate => ({
            ...coordinate,
            Mode: "DELETE"
        }));

        // Combining both of the arrays into one for the export
        const combinedCoordinates = [...updatedAddModeCoordinates, ...updatedDeleteModeCoordinates];

        const jsonBlob = new Blob([JSON.stringify(combinedCoordinates, null, 2)], { type: 'application/json' });
        saveAs(jsonBlob, `${fileName}_edited.json`);
    };

    return (
        <button className="export-button" onClick={exportJSON}>
            <i className="fa-solid fa-file-export fa-md"></i>Export Changes
        </button>
    );
};

export default JSONExporter;