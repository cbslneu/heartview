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
  lastValidDragEnd
) => {
  useEffect(() => {
    if (chartRef.current && chartRef.current.chart && isMarkUnusableMode) {
      const chart = chartRef.current.chart;

      const handleMouseDown = (event) => {
        if (isMarkUnusableMode) {
          event.preventDefault();
          dragStartRef.current = chart.xAxis[0].toValue(event.chartX);
          isDraggingRef.current = true;
        }
      };

      const handleMouseMove = (event) => {
        if (isMarkUnusableMode && isDraggingRef.current) {
          const dragEnd = chart.xAxis[0].toValue(event.chartX);

          // Remove previous temporary plot band
          if (dragPlotBandRef.current) {
            chart.xAxis[0].removePlotBand("draggingPlotBand");
          }

          lastValidDragEnd.current = dragEnd;

          // Add new temporary plot band while dragging
          dragPlotBandRef.current = {
            id: "draggingPlotBand",
            from: Math.min(dragStartRef.current, dragEnd),
            to: Math.max(dragStartRef.current, dragEnd),
            color: "rgba(255, 0, 0, 0.2)",
          };
          chart.xAxis[0].addPlotBand(dragPlotBandRef.current);
        }
      };

      const handleMouseUp = (event) => {
        if (isMarkUnusableMode && isDraggingRef.current) {
          let dragEnd = event.chartX ? chart.xAxis[0].toValue(event.chartX) : lastValidDragEnd.current;

          if (dragPlotBandRef.current) {
            chart.xAxis[0].removePlotBand("draggingPlotBand");
            dragPlotBandRef.current = null;
          }

          if (dragEnd !== null && dragStartRef.current !== null) {
            const newSegment = {
              segment: selectedSegment,
              from: Math.min(dragStartRef.current, dragEnd),
              to: Math.max(dragStartRef.current, dragEnd),
              editType: EDIT_TYPE_UNUSABLE,
              color: "rgba(255, 0, 0, 0.3)",
            };
            setUnusableSegments((prevSegments) => [...prevSegments, newSegment]);

            dragStartRef.current = null;
            isDraggingRef.current = false;
          }
        }
      };

      chart.container.addEventListener('mousedown', handleMouseDown);
      chart.container.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);

      return () => {
        chart.container.removeEventListener('mousedown', handleMouseDown);
        chart.container.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isMarkUnusableMode, chartRef, setUnusableSegments, selectedSegment, dragStartRef, isDraggingRef, dragPlotBandRef, lastValidDragEnd]);
};

export default useMarkingUnusableMode;