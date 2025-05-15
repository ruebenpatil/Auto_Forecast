function combineData(current, future) {
    const actual = current.rows.map(d => ({
        date: new Date(d[0]),
        actual: +d[1]
    }));

    const forecast = future.rows.map(d => ({
        date: new Date(d[0]),
        forecast: +d[1],
        lower: +d[2],
        upper: +d[3]
    }));

    return { actual, forecast };
}

function drawFutureForecastChart(containerSelector, currentData, futureData, width = 500, height = 300) {

    const dateFormat = currentData.date_format;
    const { actual, forecast } = combineData(currentData, futureData);

    const allDates = actual.map(d => d.date).concat(forecast.map(d => d.date));
    const allValues = actual.map(d => d.actual)
        .concat(forecast.flatMap(d => [d.forecast, d.lower, d.upper]));

    const margin = { top: 20, right: 20, bottom: 55, left: 30 };
    const svg = d3.select(containerSelector);
    svg.selectAll("*").remove();

    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);
    
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    const x = d3.scaleTime()
        .domain(d3.extent(allDates))
        .range([0, innerWidth]);

    const y = d3.scaleLinear()
        .domain([d3.min(allValues) - 10, d3.max(allValues) + 10])
        .range([innerHeight, 0]);

    // Axes
    g.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).tickFormat(d3.timeFormat(dateFormat)));

    g.append("g").call(d3.axisLeft(y));

    // Line generators
    const lineActual = d3.line()
        .x(d => x(d.date))
        .y(d => y(d.actual));

    const lineForecast = d3.line()
        .x(d => x(d.date))
        .y(d => y(d.forecast));

    const area = d3.area()
        .x(d => x(d.date))
        .y0(d => y(d.lower))
        .y1(d => y(d.upper));


    // Draw confidence area
    g.append("path")
        .datum(forecast)
        .attr("fill", "#DBEAFE")
        .attr("opacity", 0.6)
        .attr("d", area);

    g.append("path")
        .datum(forecast)
        .attr("fill", "#DBEAFE")
        .attr("opacity", 0.6)
        .attr("d", area);

    // Draw actual line
    g.append("path")
        .datum(actual)
        .attr("fill", "none")
        .attr("stroke", "#1D4ED8")
        .attr("stroke-width", 2)
        .attr("d", lineActual);

    // Draw forecast line
    g.append("path")
        .datum(forecast)
        .attr("fill", "none")
        .attr("stroke", "#EA580C")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5 3")
        .attr("d", lineForecast);



    // Add legend
    const legend = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${margin.left}, ${innerHeight + margin.top + 50})`);

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
