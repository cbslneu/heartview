import React, { Component } from "react";
import BeatChart from "./BeatChart.js";
import axios from "axios";
import "./styles.scss";

export default class PlotBeatSegment extends Component {
  constructor(props) {
    super(props);
    this.state = {
      fileData: [],
      segmentOptions: [],
      fileName: "",
      addBeatCoordinates: [],
      deleteBeatCoordinates: [],
      unusableBeats: [],
    };
  }

  componentDidMount() {
    this.fetchFile();
    const chart = document.getElementById("beatChart");

    if (chart) {
      chart.on("plotly_click", this.getClickCoordinates);
    }
  }

  componentWillUnmount() {
    const chart = document.getElementById("beatChart");
    if (chart) {
      chart.removeListener("plotly_click", this.getClickCoordinates);
    }
  }

  fetchFile = async () => {
    try {
      const response = await axios.get("http://localhost:3001/fetch-file");
      const { allFileData, allSavedData, segmentOptions } = response.data;

      if (allFileData.length > 0) {
        const fileName = allFileData[0].fileName;
        const jsonData = allFileData[0].data;

        this.setState({
          fileData: jsonData,
          fileName: fileName,
          segmentOptions: segmentOptions,
        });
      } else {
        console.error("No file data found.");
      }

      if (allSavedData && allSavedData.length > 0) {
        const jsonData = allSavedData[0].data;

        const addBeats = jsonData.filter((beat) => beat.editType === "ADD");
        const deleteBeats = jsonData.filter(
          (beat) => beat.editType === "DELETE"
        );
        const unusableBeats = jsonData.filter(
          (beat) => beat.editType === "UNUSABLE"
        )

        this.setState({
          addBeatCoordinates: addBeats,
          deleteBeatCoordinates: deleteBeats,
          unusableBeats: unusableBeats
        });
      } 
    } catch (err) {
      console.error("Error fetching JSON file: ", err);
    }
  };

  render() {
    const { fileData, segmentOptions, fileName, addBeatCoordinates, deleteBeatCoordinates, unusableBeats } = this.state;

    return (
      <div className="plot-beat-segment">
        <div className="chart-buttons"></div>
        <BeatChart
          fileData={fileData}
          fileName={fileName}
          segmentOptions={segmentOptions}
          addBeats={addBeatCoordinates}
          deleteBeats={deleteBeatCoordinates}
          unusableBeats={unusableBeats}
        />
      </div>
    );
  }
}
