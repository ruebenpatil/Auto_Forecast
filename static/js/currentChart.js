function drawCurrentForecastChart(currentData, selector) {
    const dateFormat = currentData.date_format;
    const svg = d3.select(selector);
    const margin = { top: 20, right: 20, bottom: 55, left: 30 }; // more bottom space for legend

    svg.selectAll("*").remove();

    const width = 500 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const parsedData = currentData.rows.map(d => ({
        date: d3.timeParse("%Y-%m-%d")(d[0]),
        actual: +d[1],
        forecast: +d[2],
        lower: +d[3],
        upper: +d[4]
    }));

    const x = d3.scaleTime()
        .domain(d3.extent(parsedData, d => d.date))
        .range([0, width]);

    const y = d3.scaleLinear()
        .domain([
            d3.min(parsedData, d => d.lower),
            d3.max(parsedData, d => d.upper)
        ]).nice()
        .range([height, 0]);

    const lineActual = d3.line()
        .x(d => x(d.date))
        .y(d => y(d.actual));

    const lineForecast = d3.line()
        .x(d => x(d.date))
        .y(d => y(d.forecast));

    const areaCI = d3.area()
        .x(d => x(d.date))
        .y0(d => y(d.lower))
        .y1(d => y(d.upper));

    // Confidence Interval Area
    g.append("path")
        .datum(parsedData)
        .attr("d", areaCI)
        .attr("fill", "#DBEAFE")
        .attr("opacity", 0.6);

    // Actual Line
    g.append("path")
        .datum(parsedData)
        .attr("d", lineActual)
        .attr("stroke", "#1D4ED8")
        .attr("stroke-width", 2)
        .attr("fill", "none");

    // Forecast Line
    g.append("path")
        .datum(parsedData)
        .attr("d", lineForecast)
        .attr("stroke", "#EA580C")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5 3")
        .attr("fill", "none");

    // Axes
    g.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x).tickFormat(d3.timeFormat(dateFormat)));

    g.append("g")
        .call(d3.axisLeft(y));

    // LEGEND
    const legend = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${margin.left}, ${height + margin.top + 50})`);

    const legendItems = [
        { label: "Actual", color: "#1D4ED8", type: "line", dash: false },
        { label: "Forecast", color: "#EA580C", type: "line", dash: true },
        { label: "Confidence Interval", color: "#DBEAFE", type: "area" }
    ];

    const itemSpacing = 150;

    legend.selectAll("g")
        .data(legendItems)
        .enter()
        .append("g")
        .attr("transform", (d, i) => `translate(${i * itemSpacing}, 0)`)
        .each(function(d) {
            const group = d3.select(this);

            if (d.type === "line") {
                group.append("line")
                    .attr("x1", 0).attr("y1", 0)
                    .attr("x2", 20).attr("y2", 0)
                    .attr("stroke", d.color)
                    .attr("stroke-width", 2)
                    .attr("stroke-dasharray", d.dash ? "4" : "0");
            } else if (d.type === "area") {
                group.append("rect")
                    .attr("x", 0).attr("y", -7)
                    .attr("width", 20).attr("height", 14)
                    .attr("fill", d.color)
                    .attr("opacity", 0.6);
            }

            group.append("text")
                .attr("x", 25)
                .attr("y", 1)
                .text(d.label)
                .style("font-size", "12px")
                .attr("alignment-baseline", "middle");
        });
}
