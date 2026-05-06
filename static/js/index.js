window.HELP_IMPROVE_VIDEOJS = false;

function copyBibTeX() {
    var bibtexElement = document.getElementById('bibtex-code');
    var button = document.querySelector('.copy-bibtex-btn');
    if (!bibtexElement || !button) return;
    var copyText = button.querySelector('.copy-text');
    navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
        button.classList.add('copied');
        copyText.textContent = 'Copied';
        setTimeout(function() {
            button.classList.remove('copied');
            copyText.textContent = 'Copy';
        }, 2000);
    }).catch(function() {});
}

function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function() {
    var btn = document.querySelector('.scroll-to-top');
    if (!btn) return;
    btn.classList.toggle('visible', window.pageYOffset > 300);
});

/* ================================================================
   Chart helpers
   ================================================================ */

var PI = '\u03C0';

function slopeChart(canvasId, leftLabel, rightLabel, lines, yLabel) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return;

    var datasets = lines.map(function(line, i) {
        var ds = {
            label: line.label,
            data: line.data,
            borderColor: line.color,
            backgroundColor: line.color,
            pointBackgroundColor: line.color,
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            pointRadius: 7,
            pointHoverRadius: 10,
            borderWidth: 3,
            tension: 0,
            fill: false
        };
        if (i === 0 && lines.length > 1) {
            ds.fill = { target: '+1', above: 'rgba(37, 99, 235, 0.07)', below: 'rgba(37, 99, 235, 0.07)' };
        }
        return ds;
    });

    var speedupPlugin = {
        id: 'slopeLabels',
        afterDatasetsDraw: function(chart) {
            var ctx = chart.ctx;
            chart.data.datasets.forEach(function(dataset, i) {
                var meta = chart.getDatasetMeta(i);
                if (meta.data.length < 2) return;
                var p0 = meta.data[0];
                var p1 = meta.data[1];

                ctx.save();
                ctx.textAlign = 'center';

                ctx.font = '600 13px Inter, sans-serif';
                ctx.fillStyle = dataset.borderColor;
                ctx.fillText(dataset.data[0].toLocaleString(), p0.x, p0.y - 10);
                ctx.fillText(dataset.data[1].toLocaleString(), p1.x, p1.y - 10);

                var midX = (p0.x + p1.x) / 2;
                var midY = (p0.y + p1.y) / 2;
                var tag = lines[i].speedup;
                if (tag) {
                    ctx.font = 'bold 15px Inter, sans-serif';
                    var pad = 6, tw = ctx.measureText(tag).width;
                    ctx.fillStyle = 'rgba(255,255,255,0.85)';
                    ctx.beginPath();
                    ctx.roundRect(midX - tw / 2 - pad, midY - 22, tw + pad * 2, 24, 4);
                    ctx.fill();
                    ctx.fillStyle = dataset.borderColor;
                    ctx.fillText(tag, midX, midY - 4);
                }
                ctx.restore();
            });
        }
    };

    new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels: [leftLabel, rightLabel], datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { top: 20 } },
            scales: {
                y: {
                    type: 'logarithmic',
                    min: 0,
                    title: { display: true, text: yLabel, font: { weight: 'bold' } },
                    ticks: {
                        callback: function(v) {
                            if ([0, 200, 500, 1000, 1800].indexOf(v) !== -1) return v.toLocaleString();
                            return '';
                        }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 14, weight: '600' } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            return ctx.dataset.label + ': ' + ctx.parsed.y.toLocaleString() + ' actions/sec';
                        }
                    }
                }
            }
        },
        plugins: [speedupPlugin]
    });
}

function sideBySideBar(canvasId, labels, datasets, yLabel) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return;

    new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: yLabel, font: { weight: 'bold' } }
                },
                x: {
                    ticks: { autoSkip: false, maxRotation: 25, minRotation: 0 }
                }
            },
            plugins: {
                legend: {
                    display: datasets.length > 1,
                    position: 'top'
                }
            }
        }
    });
}

/* ================================================================
   Build all charts on DOM ready
   ================================================================ */
function setupCharts() {

    /* ---- Inference Frequency ---- */
    sideBySideBar(
        'inference-chart',
        [PI + '0.5 (JAX)', PI + '0.5 (PyTorch)', PI + '0.5 (Triton)', PI + '-IMLE (Ours)'],
        [{
            label: 'Inference Frequency (Hz)',
            data: [15, 20, 25, 55],
            backgroundColor: ['#94a3b8', '#94a3b8', '#94a3b8', '#2563eb'],
            borderRadius: 6
        }],
        'Inference Freq. (Hz)'
    );

    /* ---- Action Throughput (slope chart) ---- */
    slopeChart(
        'throughput-chart',
        PI + '0.5',
        PI + '-IMLE',
        [
            { label: '5.5\u00d7 speedup (H=8 \u2192 H=12)', data: [120, 660], color: '#60a5fa', speedup: '5.5\u00d7' },
            { label: '11\u00d7 speedup (H=10 \u2192 H=30)', data: [150, 1650], color: '#1d4ed8', speedup: '11\u00d7' }
        ],
        'Actions / sec'
    );

}

$(document).ready(function() {
    setupCharts();
});
