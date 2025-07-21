import streamlit.components.v1 as components

def render_network_visualization(nodes_data, edges_data, height=500):
    """
    Render an interactive D3.js network visualization
    
    Args:
        nodes_data: List of dicts with node properties
        edges_data: List of dicts with edge properties
        height: Height of the visualization
    """
    
    html_content = f"""
    <div id="network-viz" style="width: 100%; height: {height}px;"></div>
    
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    // Data from Python
    const nodes = {nodes_data};
    const links = {edges_data};
    
    // Set dimensions
    const width = document.getElementById('network-viz').offsetWidth;
    const height = {height};
    
    // Create SVG
    const svg = d3.select("#network-viz")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .style("background", "rgba(0, 0, 0, 0.1)")
        .style("border-radius", "10px");
    
    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(50))
        .force("charge", d3.forceManyBody().strength(-100))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(20));
    
    // Create links
    const link = svg.append("g")
        .selectAll("line")
        .data(links)
        .enter().append("line")
        .style("stroke", "rgba(255, 255, 255, 0.2)")
        .style("stroke-width", 2);
    
    // Create nodes
    const node = svg.append("g")
        .selectAll("circle")
        .data(nodes)
        .enter().append("circle")
        .attr("r", d => d.influence * 10 + 5)
        .style("fill", d => d.party === "Opposition" ? "#00ff88" : "#ff0088")
        .style("stroke", "#fff")
        .style("stroke-width", 2)
        .style("cursor", "pointer")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));
    
    // Add labels
    const label = svg.append("g")
        .selectAll("text")
        .data(nodes)
        .enter().append("text")
        .text(d => d.name)
        .style("font-size", "12px")
        .style("fill", "white")
        .style("text-anchor", "middle")
        .style("pointer-events", "none");
    
    // Add hover effects
    node.on("mouseover", function(event, d) {{
        d3.select(this)
            .transition()
            .duration(200)
            .attr("r", d.influence * 15 + 8)
            .style("filter", "drop-shadow(0 0 10px currentColor)");
        
        // Show tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("position", "absolute")
            .style("background", "rgba(0, 0, 0, 0.8)")
            .style("color", "white")
            .style("padding", "10px")
            .style("border-radius", "5px")
            .style("pointer-events", "none")
            .style("opacity", 0);
        
        tooltip.transition()
            .duration(200)
            .style("opacity", 0.9);
        
        tooltip.html(`
            <strong>${{d.name}}</strong><br/>
            Party: ${{d.party}}<br/>
            Influence: ${{(d.influence * 100).toFixed(0)}}%<br/>
            Connections: ${{links.filter(l => l.source.id === d.id || l.target.id === d.id).length}}
        `)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px");
    }})
    .on("mouseout", function(event, d) {{
        d3.select(this)
            .transition()
            .duration(200)
            .attr("r", d.influence * 10 + 5)
            .style("filter", "none");
        
        d3.selectAll(".tooltip").remove();
    }});
    
    // Animation tick
    simulation.on("tick", () => {{
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
        
        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
        
        label
            .attr("x", d => d.x)
            .attr("y", d => d.y - 15);
    }});
    
    // Drag functions
    function dragstarted(event, d) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }}
    
    function dragged(event, d) {{
        d.fx = event.x;
        d.fy = event.y;
    }}
    
    function dragended(event, d) {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }}
    
    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.5, 3])
        .on("zoom", (event) => {{
            svg.selectAll("g").attr("transform", event.transform);
        }});
    
    svg.call(zoom);
    
    // Animate entrance
    node
        .attr("r", 0)
        .transition()
        .duration(1000)
        .delay((d, i) => i * 50)
        .attr("r", d => d.influence * 10 + 5);
    
    </script>
    
    <style>
    .tooltip {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }}
    </style>
    """
    
    components.html(html_content, height=height)

def render_animated_counter(target_value, label, duration=2000):
    """
    Render an animated counter with easing
    
    Args:
        target_value: Final value to count to
        label: Label for the counter
        duration: Animation duration in milliseconds
    """
    
    html_content = f"""
    <div id="counter-container" style="text-align: center; padding: 20px;">
        <h2 style="color: #00ff88; font-size: 48px; margin: 0;">
            <span id="counter">0</span>
        </h2>
        <p style="color: #888; font-size: 18px; margin: 5px 0;">{label}</p>
    </div>
    
    <script>
    // Easing function for smooth animation
    function easeOutQuart(t) {{
        return 1 - Math.pow(1 - t, 4);
    }}
    
    // Animate counter
    function animateCounter(start, end, duration) {{
        const startTime = performance.now();
        const counter = document.getElementById('counter');
        
        function update(currentTime) {{
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Apply easing
            const easedProgress = easeOutQuart(progress);
            const currentValue = Math.floor(start + (end - start) * easedProgress);
            
            // Format with commas
            counter.textContent = currentValue.toLocaleString();
            
            // Add glow effect during animation
            const glowIntensity = Math.sin(progress * Math.PI) * 20;
            counter.style.textShadow = `0 0 ${{glowIntensity}}px #00ff88`;
            
            if (progress < 1) {{
                requestAnimationFrame(update);
            }} else {{
                // Final glow pulse
                counter.style.transition = 'text-shadow 0.5s ease';
                counter.style.textShadow = '0 0 30px #00ff88';
                setTimeout(() => {{
                    counter.style.textShadow = '0 0 10px #00ff88';
                }}, 500);
            }}
        }}
        
        requestAnimationFrame(update);
    }}
    
    // Start animation
    animateCounter(0, {target_value}, {duration});
    </script>
    """
    
    components.html(html_content, height=150)

def render_particle_visualization(particle_count=100):
    """
    Render an animated particle system visualization
    """
    
    html_content = f"""
    <canvas id="particles" style="width: 100%; height: 300px; border-radius: 10px;"></canvas>
    
    <script>
    const canvas = document.getElementById('particles');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = 300;
    
    // Particle class
    class Particle {{
        constructor() {{
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 2;
            this.vy = (Math.random() - 0.5) * 2;
            this.radius = Math.random() * 3 + 1;
            this.color = Math.random() > 0.5 ? '#00ff88' : '#0088ff';
            this.connections = [];
        }}
        
        update() {{
            this.x += this.vx;
            this.y += this.vy;
            
            // Bounce off walls
            if (this.x < 0 || this.x > canvas.width) this.vx = -this.vx;
            if (this.y < 0 || this.y > canvas.height) this.vy = -this.vy;
        }}
        
        draw() {{
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.fill();
            
            // Glow effect
            ctx.shadowBlur = 10;
            ctx.shadowColor = this.color;
            ctx.fill();
            ctx.shadowBlur = 0;
        }}
    }}
    
    // Create particles
    const particles = [];
    for (let i = 0; i < {particle_count}; i++) {{
        particles.push(new Particle());
    }}
    
    // Animation loop
    function animate() {{
        ctx.fillStyle = 'rgba(15, 15, 30, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Update and draw particles
        particles.forEach(particle => {{
            particle.update();
            particle.draw();
        }});
        
        // Draw connections
        particles.forEach((p1, i) => {{
            particles.slice(i + 1).forEach(p2 => {{
                const distance = Math.sqrt(
                    Math.pow(p1.x - p2.x, 2) + 
                    Math.pow(p1.y - p2.y, 2)
                );
                
                if (distance < 100) {{
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(255, 255, 255, ${{0.2 * (1 - distance / 100)}})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }}
            }});
        }});
        
        requestAnimationFrame(animate);
    }}
    
    animate();
    </script>
    """
    
    components.html(html_content, height=300)

# Usage example in main app:
"""
# In your Streamlit app:
from custom_viz_component import render_network_visualization, render_animated_counter

# Example network data
nodes = [
    {"id": 1, "name": "Kampala", "party": "Opposition", "influence": 0.8},
    {"id": 2, "name": "Wakiso", "party": "Ruling", "influence": 0.6},
    {"id": 3, "name": "Mukono", "party": "Opposition", "influence": 0.7},
]

edges = [
    {"source": 1, "target": 2},
    {"source": 2, "target": 3},
    {"source": 1, "target": 3},
]

# Render the visualization
render_network_visualization(nodes, edges)

# Animated counter
render_animated_counter(1234567, "Total Registered Voters")
"""
