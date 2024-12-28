const createChartOptions = ({
    xAxisData,
    initECGsData,
    initBeats,
    initArtifacts,
    addModeCoordinates,
    deleteModeCoordinates,
    selectedSegment,
    unusableSegments,
    isAddMode,
    isDeleteMode,
    isMarkingUnusableMode,
    handleChartClick,
    dataTypeX,
    isPanningRef
  }) => {
    return {
      chart: {
        type: "line",
        zoomType: isMarkingUnusableMode ? null : "x",
        panning: !isMarkingUnusableMode, // Enable panning
        panKey: "shift", // Optional: Hold Shift to pan
        events: {
          pan: (event) => {
            if (event.shiftKey) {
                isPanningRef.current = true;
            }
          },
          click: function (event) {
            if ((isAddMode || isDeleteMode) && !event.shiftKey) {
              handleChartClick(event);
            }
          },
        },
        style: {
          fontFamily: "'Poppins', sans-serif",
          fontSize: "20px",
        },
        animation: false,
      },
      title: {
        text: "",
      },
      xAxis: {
        title: {
          text: dataTypeX,
        },
        labels: {
          formatter: function () {
            const date = new Date(this.value);
            return date.toUTCString().split(" ")[4];
          },
          style: {
            fontSize: "13px",
          },
        },
        allowDecimals: true,
        plotBands: unusableSegments,
        min: xAxisData[0],
        max: xAxisData[xAxisData.length - 1],
      },
      yAxis: {
        title: {
          text: "Signal",
        },
        allowDecimals: true,
      },
      tooltip: {
        formatter: function () {
          const date = new Date(this.x);
          return `<b>${this.series.name}</b><br/>Time: ${
            date.toUTCString().split(" ")[4]
          } <br/>Amplitude: ${this.y.toFixed(3)} mV`;
        },
      },
      series: [
        {
          name: "ECG",
          data: initECGsData,
          color: "#3562BD",
          turboThreshold: 0,
          states: {
            hover: {
              enabled: false,
            },
            inactive: {
              enabled: false,
            },
          },
          point: {
            events: {
              click: isAddMode || isDeleteMode ? handleChartClick : "",
            },
          },
        },
        {
          name: "Beat",
          data: initBeats,
          type: "scatter",
          color: "#F9C669",
          marker: {
            symbol: "circle",
          },
          turboThreshold: 0,
          states: {
            hover: {
              enabled: false,
            },
            inactive: {
              enabled: false,
            },
          },
          point: {
            events: {
              click: isAddMode || isDeleteMode ? handleChartClick : "",
            },
          },
        },
        {
          name: "Artifact",
          data: initArtifacts,
          type: "scatter",
          color: "red",
          marker: {
            symbol: "circle",
          },
          visible: initArtifacts.length > 0,
          showInLegend: initArtifacts.length > 0,
          turboThreshold: 0,
          states: {
            hover: {
              enabled: false,
            },
          },
          point: {
            events: {
              click: isAddMode || isDeleteMode ? handleChartClick : "",
            },
          },
        },
        {
          name: "Added Beats",
          data: addModeCoordinates.filter((o) => o.segment === selectedSegment),
          type: "scatter",
          color: "#02E337",
          marker: {
            symbol: "circle",
          },
          visible: addModeCoordinates.some((o) => o.segment === selectedSegment),
          showInLegend: addModeCoordinates.some(
            (o) => o.segment === selectedSegment
          ),
          turboThreshold: 0,
          states: {
            hover: {
              enabled: false,
            },
            inactive: {
              enabled: false,
            },
          },
          point: {
            events: {
              click: handleChartClick,
            },
          },
        },
        {
          name: "Deleted Beats",
          data: deleteModeCoordinates.filter(
            (o) => o.segment === selectedSegment
          ),
          type: "scatter",
          color: "red",
          marker: {
            symbol: "cross",
            lineColor: null,
            lineWidth: 2,
          },
          visible: deleteModeCoordinates.some(
            (o) => o.segment === selectedSegment
          ),
          showInLegend: deleteModeCoordinates.some(
            (o) => o.segment === selectedSegment
          ),
          turboThreshold: 0,
          states: {
            hover: {
              enabled: false,
            },
            inactive: {
              enabled: false,
            },
          },
          point: {
            events: {
              click: handleChartClick,
            },
          },
        },
      ],
    };
  };
  
  export default createChartOptions;
  