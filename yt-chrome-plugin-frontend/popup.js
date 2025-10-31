

// popup.js - FIXED CORRELATION ANALYSIS

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  let API_KEY = null;
  const API_URL = 'http://localhost:8080';

  // --- HELPER FUNCTIONS ---

  // Parse ISO 8601 duration (e.g., PT5M30S) to seconds
  function parseISO8601Duration(duration) {
    const match = duration.match(/PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/);
    if (!match) return 0;
    const hours = parseInt(match[1] || 0);
    const minutes = parseInt(match[2] || 0);
    const seconds = parseInt(match[3] || 0);
    return hours * 3600 + minutes * 60 + seconds;
  }

  // Format seconds to readable duration
  function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  }

  // Fetch video metadata (duration and publish time)
  async function fetchVideoMetadata(videoId) {
    try {
      const response = await fetch(
        `https://www.googleapis.com/youtube/v3/videos?part=contentDetails,snippet&id=${videoId}&key=${API_KEY}`
      );
      const data = await response.json();
      if (data.items && data.items.length > 0) {
        const item = data.items[0];
        return {
          duration: parseISO8601Duration(item.contentDetails.duration),
          publishedAt: new Date(item.snippet.publishedAt)
        };
      }
      return { duration: 0, publishedAt: null };
    } catch (error) {
      console.error("Error fetching video metadata:", error);
      return { duration: 0, publishedAt: null };
    }
  }

  // Calculate Pearson correlation coefficient
  function correlation(x, y) {
    if (x.length !== y.length || x.length < 2) return 0;
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((a, b, i) => a + b * y[i], 0);
    const sumX2 = x.reduce((a, b) => a + b * b, 0);
    const sumY2 = y.reduce((a, b) => a + b * b, 0);
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX ** 2) * (n * sumY2 - sumY ** 2));
    return denominator === 0 ? 0 : numerator / denominator;
  }

  // Analyze sentiment trends over time
  function analyzeSentimentTrend(predictions, videoPublishDate) {
    if (!videoPublishDate || predictions.length < 2) {
      return { 
        correlation: 0, 
        interpretation: "Not enough data", 
        trend: "N/A",
        earlyAvg: 0,
        lateAvg: 0,
        change: 0
      };
    }

    // Sort predictions by timestamp
    const sorted = predictions
      .map(p => ({
        sentiment: parseInt(p.sentiment),
        timestamp: new Date(p.timestamp),
        hoursAfterPublish: (new Date(p.timestamp) - videoPublishDate) / (1000 * 60 * 60)
      }))
      .sort((a, b) => a.hoursAfterPublish - b.hoursAfterPublish);

    // Extract data for correlation
    const timePoints = sorted.map(s => s.hoursAfterPublish);
    const sentiments = sorted.map(s => s.sentiment);

    // Calculate correlation
    const corr = correlation(timePoints, sentiments);

    // Calculate early vs late averages
    const thirdPoint = Math.floor(sorted.length / 3);
    const earlyComments = sorted.slice(0, thirdPoint);
    const lateComments = sorted.slice(-thirdPoint);
    
    const earlyAvg = earlyComments.reduce((sum, c) => sum + c.sentiment, 0) / earlyComments.length;
    const lateAvg = lateComments.reduce((sum, c) => sum + c.sentiment, 0) / lateComments.length;
    
    // Normalize to 0-10 scale
    const earlyNormalized = ((earlyAvg + 1) / 2) * 10;
    const lateNormalized = ((lateAvg + 1) / 2) * 10;
    const change = lateNormalized - earlyNormalized;
    const changePercent = (change / earlyNormalized) * 100;

    // Improved interpretation using BOTH correlation and actual change
    let interpretation = "";
    let trend = "";
    
    // Use change magnitude as primary indicator
    if (Math.abs(change) < 0.5) {
      // Very small change (< 0.5 points) = Stable
      interpretation = "Sentiment remains consistent";
      trend = "➡️ Stable";
    } else if (change > 0.5) {
      // Positive change
      if (change > 1.5) {
        interpretation = "Sentiment significantly improves over time";
        trend = "📈 Strong Positive Trend";
      } else {
        interpretation = "Sentiment improves over time";
        trend = "📈 Positive Trend";
      }
    } else {
      // Negative change
      if (change < -1.5) {
        interpretation = "Sentiment significantly declines over time";
        trend = "📉 Strong Negative Trend";
      } else {
        interpretation = "Sentiment slightly declines over time";
        trend = "📉 Declining";
      }
    }

    // Add warning for significant drops
    if (changePercent < -10) {
      interpretation += ` (${Math.abs(changePercent).toFixed(1)}% drop)`;
    } else if (changePercent > 10) {
      interpretation += ` (${changePercent.toFixed(1)}% increase)`;
    }

    return { 
      correlation: corr, 
      interpretation, 
      trend,
      earlyComments,
      lateComments,
      earlyAvg: earlyNormalized,
      lateAvg: lateNormalized,
      change,
      changePercent
    };
  }

  // Calculate engagement metrics
  function calculateEngagementMetrics(comments, predictions, videoDuration) {
    const avgSentiment = predictions.reduce((sum, p) => sum + parseInt(p.sentiment), 0) / predictions.length;
    const normalizedSentiment = ((avgSentiment + 1) / 2) * 10;
    
    // Estimate engagement score (combining sentiment and comment density)
    const commentDensity = comments.length / (videoDuration / 60); // comments per minute
    const engagementScore = (normalizedSentiment * 0.7 + Math.min(commentDensity * 2, 10) * 0.3).toFixed(2);
    
    return {
      avgSentiment: normalizedSentiment.toFixed(2),
      commentDensity: commentDensity.toFixed(2),
      engagementScore
    };
  }

  // --- FETCH API KEY ---
  try {
    outputDiv.innerHTML = "<p>Initializing...</p>";
    const keyResponse = await fetch(`${API_URL}/get_youtube_api_key`);
    if (!keyResponse.ok) throw new Error('Failed to fetch API key');
    const keyData = await keyResponse.json();
    API_KEY = keyData.api_key;
    if (!API_KEY) throw new Error('API key not found');
  } catch (error) {
    console.error("Error fetching API key:", error);
    outputDiv.innerHTML = `
      <div style="color: #ff6b6b; padding: 10px;">
        <p><strong>Configuration Error</strong></p>
        <p>Unable to fetch YouTube API key.</p>
      </div>`;
    return;
  }

  // --- GET VIDEO ID ---
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (!match || !match[1]) {
      outputDiv.innerHTML = "<p>This is not a valid YouTube URL.</p>";
      return;
    }

    const videoId = match[1];
    outputDiv.innerHTML = `<p>Video ID: ${videoId}</p><p>Fetching video metadata and comments...</p>`;

    // --- FETCH VIDEO METADATA ---
    const videoMetadata = await fetchVideoMetadata(videoId);
    const videoDuration = videoMetadata.duration;
    const videoPublishDate = videoMetadata.publishedAt;

    // --- FETCH COMMENTS ---
    const comments = await fetchComments(videoId);
    if (comments.length === 0) {
      outputDiv.innerHTML += "<p>No comments found.</p>";
      return;
    }

    outputDiv.innerHTML += `<p>Fetched ${comments.length} comments. Analyzing sentiment...</p>`;
    const predictions = await getSentimentPredictions(comments);
    if (!predictions) return;

    // --- PROCESS SENTIMENTS ---
    const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
    const sentimentData = [];
    predictions.forEach(item => {
      sentimentCounts[item.sentiment]++;
      sentimentData.push({ 
        timestamp: item.timestamp, 
        sentiment: parseInt(item.sentiment) 
      });
    });

    // --- CALCULATE METRICS ---
    const totalComments = comments.length;
    const uniqueCommenters = new Set(comments.map(c => c.authorId)).size;
    const totalWords = comments.reduce((sum, c) => 
      sum + c.text.split(/\s+/).filter(w => w.length > 0).length, 0
    );
    const avgWordLength = (totalWords / totalComments).toFixed(2);

    // Calculate engagement metrics
    const metrics = calculateEngagementMetrics(comments, predictions, videoDuration);

    // Analyze sentiment trend
    const trendAnalysis = analyzeSentimentTrend(predictions, videoPublishDate);

    // Build early vs late text with visual indicator
    let earlyVsLateText = "N/A";
    let trendIndicator = "";
    if (trendAnalysis.earlyAvg && trendAnalysis.lateAvg) {
      const arrow = trendAnalysis.change > 0.5 ? "📈" : 
                   trendAnalysis.change < -0.5 ? "📉" : "➡️";
      earlyVsLateText = `${arrow} ${trendAnalysis.earlyAvg.toFixed(1)}/10 → ${trendAnalysis.lateAvg.toFixed(1)}/10`;
      
      // Add actionable insight
      if (trendAnalysis.change < -0.5) {
        trendIndicator = `⚠️ Viewers become less satisfied over time`;
      } else if (trendAnalysis.change > 0.5) {
        trendIndicator = `✅ Viewers become more satisfied over time`;
      } else {
        trendIndicator = `✓ Consistent viewer satisfaction`;
      }
    }

    // --- DISPLAY ENHANCED SUMMARY ---
    outputDiv.innerHTML = `
      <div class="section">
        <div class="section-title">Comment Analysis Summary</div>
        <div class="metrics-container">
          <div class="metric">
            <div class="metric-title">Total Comments</div>
            <div class="metric-value">${totalComments}</div>
          </div>
          <div class="metric">
            <div class="metric-title">Unique Commenters</div>
            <div class="metric-value">${uniqueCommenters}</div>
          </div>
          <div class="metric">
            <div class="metric-title">Avg Comment Length</div>
            <div class="metric-value">${avgWordLength} words</div>
          </div>
          <div class="metric">
            <div class="metric-title">Video Duration</div>
            <div class="metric-value">${formatDuration(videoDuration)}</div>
          </div>
          <div class="metric">
            <div class="metric-title">Avg Sentiment Score</div>
            <div class="metric-value">${metrics.avgSentiment}/10</div>
          </div>
          <div class="metric">
            <div class="metric-title">Engagement Score</div>
            <div class="metric-value">${metrics.engagementScore}/10</div>
          </div>
          <div class="metric">
            <div class="metric-title">Comment Density</div>
            <div class="metric-value">${metrics.commentDensity}/min</div>
          </div>
          <div class="metric">
            <div class="metric-title">Sentiment Trend</div>
            <div class="metric-value">${trendAnalysis.trend}</div>
          </div>
          <div class="metric" style="width: 100%;">
            <div class="metric-title">Sentiment Over Time</div>
            <div class="metric-value">${earlyVsLateText}</div>
          </div>
          <div class="metric" style="width: 100%;">
            <div class="metric-title">Correlation: Time ↔ Sentiment</div>
            <div class="metric-value">${trendAnalysis.correlation.toFixed(2)} (${trendAnalysis.interpretation})</div>
          </div>
        </div>
      </div>

      <div class="section" style="background-color: #2a2a2a; padding: 15px; border-radius: 8px; margin: 15px 0;">
        <div class="section-title" style="color: #0099ff; margin-bottom: 10px;">📊 Key Insights</div>
        <div style="color: #f1f1f1; line-height: 1.8;">
          <p style="margin: 5px 0;">${trendIndicator}</p>
          ${trendAnalysis.change < -0.5 ? `
            <p style="margin: 5px 0; color: #ff6b6b;">
              💡 <strong>Content Recommendation:</strong> Viewers who watch longer become less satisfied. 
              Consider shortening the video or adding more engaging content in the second half.
            </p>
          ` : trendAnalysis.change > 0.5 ? `
            <p style="margin: 5px 0; color: #51cf66;">
              ✨ <strong>Great Job!</strong> Your content quality improves over time or builds momentum. 
              Viewers who stick around are more satisfied!
            </p>
          ` : `
            <p style="margin: 5px 0; color: #74c0fc;">
              ✓ <strong>Consistent Quality:</strong> Your content maintains stable sentiment throughout. 
              Viewers have a consistent experience from start to finish.
            </p>
          `}
          ${metrics.commentDensity < 1 ? `
            <p style="margin: 5px 0; color: #ffd43b;">
              📢 <strong>Engagement Tip:</strong> Low comment density (${metrics.commentDensity}/min). 
              Consider adding discussion prompts or questions to encourage more interaction.
            </p>
          ` : metrics.commentDensity > 3 ? `
            <p style="margin: 5px 0; color: #51cf66;">
              🔥 <strong>High Engagement:</strong> ${metrics.commentDensity} comments/min shows strong viewer interaction!
            </p>
          ` : ''}
          ${parseFloat(metrics.avgSentiment) < 6 ? `
            <p style="margin: 5px 0; color: #ff6b6b;">
              ⚠️ <strong>Watch Alert:</strong> Below-average sentiment (${metrics.avgSentiment}/10). 
              Review negative comments for actionable feedback.
            </p>
          ` : parseFloat(metrics.avgSentiment) > 8 ? `
            <p style="margin: 5px 0; color: #51cf66;">
              🌟 <strong>Excellent Reception:</strong> ${metrics.avgSentiment}/10 sentiment! 
              Your audience loves this content.
            </p>
          ` : ''}
        </div>
      </div>

      <div class="section">
        <div class="section-title">Sentiment Analysis Results</div>
        <p>See the pie chart below for sentiment distribution.</p>
        <div id="chart-container"></div>
      </div>

      <div class="section">
        <div class="section-title">Sentiment Trend Over Time</div>
        <div id="trend-graph-container"></div>
      </div>

      <div class="section">
        <div class="section-title">Comment Wordcloud</div>
        <div id="wordcloud-container"></div>
      </div>
    `;

    // --- DISPLAY CHARTS ---
    await fetchAndDisplayChart(sentimentCounts);
    await fetchAndDisplayTrendGraph(sentimentData);
    await fetchAndDisplayWordCloud(comments.map(c => c.text));

    // --- DISPLAY TOP COMMENTS ---
    outputDiv.innerHTML += `
      <div class="section">
        <div class="section-title">Top 25 Comments with Sentiments</div>
        <ul class="comment-list">
          ${predictions.slice(0, 25).map((item, i) => `
            <li class="comment-item">
              <span>${i + 1}. ${item.comment}</span><br>
              <span class="comment-sentiment">Sentiment: ${item.sentiment}</span>
            </li>`).join('')}
        </ul>
      </div>`;
  });

  // --- HELPER FUNCTIONS ---
  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 500) {
        const response = await fetch(
          `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`
        );
        const data = await response.json();
        if (data.error) {
          console.error("YouTube API error:", data.error);
          break;
        }
        if (data.items) {
          data.items.forEach(item => {
            const snippet = item.snippet.topLevelComment.snippet;
            comments.push({
              text: snippet.textOriginal,
              timestamp: snippet.publishedAt,
              authorId: snippet.authorChannelId?.value || 'Unknown'
            });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      const result = await response.json();
      if (response.ok) return result;
      throw new Error(result.error || 'Error fetching predictions');
    } catch (error) {
      console.error("Error fetching predictions:", error);
      return null;
    }
  }

  async function fetchAndDisplayChart(sentimentCounts) {
    try {
      const response = await fetch(`${API_URL}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts })
      });
      if (!response.ok) throw new Error('Failed to fetch chart');
      const blob = await response.blob();
      const img = document.createElement('img');
      img.src = URL.createObjectURL(blob);
      img.style.width = '100%';
      document.getElementById('chart-container')?.appendChild(img);
    } catch (error) {
      console.error(error);
    }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const response = await fetch(`${API_URL}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData })
      });
      if (!response.ok) throw new Error('Failed to fetch trend graph');
      const blob = await response.blob();
      const img = document.createElement('img');
      img.src = URL.createObjectURL(blob);
      img.style.width = '100%';
      document.getElementById('trend-graph-container')?.appendChild(img);
    } catch (error) {
      console.error(error);
    }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      if (!response.ok) throw new Error('Failed to fetch word cloud');
      const blob = await response.blob();
      const img = document.createElement('img');
      img.src = URL.createObjectURL(blob);
      img.style.width = '100%';
      document.getElementById('wordcloud-container')?.appendChild(img);
    } catch (error) {
      console.error(error);
    }
  }
});