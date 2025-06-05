import { useEffect } from "react";
import { EDIT_TYPE_UNUSABLE } from "../constants/constants";

const useMarkingUnusableMode = (
  isMarkUnusableMode,
  chartRef,
  setUnusableSegments,
  selectedSegment,
  dragStartRef,
  isDraggingRef,
  dragPlotBandRef,
  lastValidDragEnd,
  segmentBoundaries
) => {
  useEffect(() => {
    if (chartRef.current && chartRef.current.chart && isMarkUnusableMode) {
      const chart = chartRef.current.chart;

      const handleMouseDown = (event) => {
        if (isMarkUnusableMode) {
          event.preventDefault();

          let dragStart = chart.xAxis[0].toValue(event.chartX);
          if (!isNaN(dragStart)) {
            dragStartRef.current = dragStart;
            isDraggingRef.current = true;
          } 
        }
      };

      const handleMouseMove = (event) => {
        if (isMarkUnusableMode && isDraggingRef.current) {
          let dragEnd = chart.xAxis[0].toValue(event.chartX);

          if (!isNaN(dragEnd)) {
            // Remove previous temporary plot band
            if (dragPlotBandRef.current) {
              chart.xAxis[0].removePlotBand("draggingPlotBand");
            }

            // Add new temporary plot band while dragging
            lastValidDragEnd.current = dragEnd;
            dragPlotBandRef.current = {
              id: "draggingPlotBand",
              from: Math.min(dragStartRef.current, dragEnd),
              to: Math.max(dragStartRef.current, dragEnd),
              color: "rgba(255, 0, 0, 0.2)",
            };
            chart.xAxis[0].addPlotBand(dragPlotBandRef.current);
          } 
        }
      };

      const handleMouseUp = (event) => {
        if (isMarkUnusableMode && isDraggingRef.current) {
          let dragEnd =
            lastValidDragEnd.current > segmentBoundaries.to.x
              ? segmentBoundaries.to.x
              : lastValidDragEnd.current;

          // Clamp drag end within segment boundaries
          dragEnd = Math.max(
            segmentBoundaries.from.x,
            Math.min(segmentBoundaries.to.x, dragEnd)
          );

          if (!isNaN(dragStartRef.current) && !isNaN(dragEnd)) {
            // Remove temporary plot band
            if (dragPlotBandRef.current) {
              chart.xAxis[0].removePlotBand("draggingPlotBand");
              dragPlotBandRef.current = null;
            }

            // Add final unusable segment
            const newSegment = {
              segment: selectedSegment,
              from: Math.min(dragStartRef.current, dragEnd),
              to: Math.max(dragStartRef.current, dragEnd),
              editType: EDIT_TYPE_UNUSABLE,
              color: "rgba(255, 0, 0, 0.3)",
            };

            setUnusableSegments((prevSegments) => [
              ...prevSegments,
              newSegment,
            ]);
          }

          // Reset dragging state
          dragStartRef.current = null;
          isDraggingRef.current = false;
        }
      };

      // Attach event listeners to the chart container
      chart.container.addEventListener("mousedown", handleMouseDown);
      chart.container.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);

      return () => {
        // Cleanup temporary plot bands if any
        if (dragPlotBandRef.current) {
          chart.xAxis[0].removePlotBand("draggingPlotBand");
          dragPlotBandRef.current = null;
        }

        // Cleanup event listeners
        chart.container.removeEventListener("mousedown", handleMouseDown);
        chart.container.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);

        dragStartRef.current = null;
        isDraggingRef.current = false;
        lastValidDragEnd.current = null;
      };
    }
  }, [
    isMarkUnusableMode,
    chartRef,
    setUnusableSegments,
    selectedSegment,
    dragStartRef,
    isDraggingRef,
    dragPlotBandRef,
    lastValidDragEnd,
    segmentBoundaries,
  ]);
};

export default useMarkingUnusableMode;