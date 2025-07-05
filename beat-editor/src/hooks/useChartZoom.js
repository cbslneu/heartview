import { useEffect } from "react";

const useChartZoom = (chartRef, chartOptions) => {
  useEffect(() => {
    const handleScrollZoom = (event) => {
      if (chartRef.current) {
        const chart = chartRef.current.chart;
        event.preventDefault();

        const zoomFactor = event.deltaY < 0 ? 0.9 : 1.05; // Smoother zooming
        const xAxis = chart.xAxis[0];
        const yAxis = chart.yAxis[0];

        // Get the current extremes (current view range)
        const minX = xAxis.min;
        const maxX = xAxis.max;
        const minY = yAxis.min;
        const maxY = yAxis.max;

        // Get the data's full range across all series (ECG, Beats, Artifacts)
        let allXValues = [];
        let allYValues = [];

        chart.series.forEach((series) => {
          series.data.forEach((point) => {
            allXValues.push(point.x);
            allYValues.push(point.y);
          });
        });

        const originalMinX = Math.min(...allXValues);
        const originalMaxX = Math.max(...allXValues);
        const originalMinY = Math.min(...allYValues);
        const originalMaxY = Math.max(...allYValues);

        // Find the mouse position in axis values
        const mouseX = xAxis.toValue(event.offsetX);
        const mouseY = yAxis.toValue(event.offsetY);

        console.log("Mouse position:", { mouseX, mouseY });

        // Calculate new min and max values based on the mouse position
        const newMinX = Math.max(
          mouseX - (mouseX - minX) * zoomFactor,
          originalMinX
        );
        const newMaxX = Math.min(
          mouseX + (maxX - mouseX) * zoomFactor,
          originalMaxX
        );
        const newMinY = Math.max(
          mouseY - (mouseY - minY) * zoomFactor,
          originalMinY
        );
        const newMaxY = Math.min(
          mouseY + (maxY - mouseY) * zoomFactor,
          originalMaxY
        );

        console.log("New extremes:", { newMinX, newMaxX, newMinY, newMaxY });

        // Set the new extremes for X and Y axes
        xAxis.setExtremes(newMinX, newMaxX);
        yAxis.setExtremes(newMinY, newMaxY);
      }
    };

    const chartContainer = document.querySelector(".highcharts-container");
    if (chartContainer) {
      chartContainer.addEventListener("wheel", handleScrollZoom, {
        passive: false,
      });
    }

    return () => {
      if (chartContainer) {
        chartContainer.removeEventListener("wheel", handleScrollZoom);
      }
    };
  }, [chartRef, chartOptions]);
};

export default useChartZoom;
