import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
import nltk


def create_word_frequency_bar_chart(word_freq: Counter, top_n: int = 20, 
                                  title: str = "Top Words Frequency") -> go.Figure:
    """
    Create a bar chart of the most frequent words
    
    Parameters:
    -----------
    word_freq : Counter
        Counter object with word frequencies
    top_n : int
        Number of top words to display
    title : str
        Chart title
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    most_common_words = word_freq.most_common(top_n)
    df_words = pd.DataFrame(most_common_words, columns=["word", "count"])
    
    fig = px.bar(df_words, x="count", y="word", orientation="h", 
                 color="count", color_continuous_scale="Viridis",
                 title=title)
    
    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="Word",
        template="plotly_dark",
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_interactive_tag_cloud(word_freq: Counter, top_n: int = 80, 
                               min_size: int = 12, max_size: int = 48) -> go.Figure:
    """
    Create an interactive tag cloud visualization
    
    Parameters:
    -----------
    word_freq : Counter
        Counter object with word frequencies
    top_n : int
        Number of top words to display
    min_size : int
        Minimum font size
    max_size : int
        Maximum font size
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    items = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, counts = zip(*items)
    dfw = pd.DataFrame({"word": words, "count": counts})
    
    sizes = np.interp(dfw["count"], (dfw["count"].min(), dfw["count"].max()), 
                     (min_size, max_size))
    
    n_cols = 6
    n_rows = int(np.ceil(len(dfw) / n_cols))
    x = np.tile(np.arange(n_cols), n_rows)[:len(dfw)]
    y = -np.repeat(np.arange(n_rows), n_cols)[:len(dfw)]
    
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="text",
        text=dfw["word"],
        textfont=dict(size=sizes),
        hovertext=[f"{w}: {c}" for w, c in zip(dfw["word"], dfw["count"])],
        hoverinfo="text"
    ))
    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title="Interactive Tag Wall (descending sizes, hover for counts)",
        plot_bgcolor="white", 
        margin=dict(t=60, l=0, r=0, b=0)
    )
    
    return fig


def get_word_frequencies(texts: list, remove_stopwords: bool = True) -> Counter:
    """
    Calculate word frequencies from a list of texts
    
    Parameters:
    -----------
    texts : list
        List of text strings
    remove_stopwords : bool
        Whether to remove stopwords
    
    Returns:
    --------
    Counter
        Word frequency counter
    """
    stop_words = set(stopwords.words("english")) if remove_stopwords else set()
    all_words = []
    
    for text in texts:
        for word in text.split():
            if word not in stop_words:
                all_words.append(word)
    
    return Counter(all_words)


def compare_word_frequencies(texts1: list, texts2: list, top_n: int = 20, 
                           title: str = "Word Frequency Comparison") -> go.Figure:
    """
    Compare word frequencies between two sets of texts
    
    Parameters:
    -----------
    texts1 : list
        First list of texts
    texts2 : list
        Second list of texts
    top_n : int
        Number of top words to compare
    title : str
        Chart title
    
    Returns:
    --------
    go.Figure
        Plotly figure with comparison
    """
    freq1 = get_word_frequencies(texts1)
    freq2 = get_word_frequencies(texts2)
    
    # Get top words from both sets combined
    all_freq = freq1 + freq2
    top_words = [word for word, _ in all_freq.most_common(top_n)]
    
    # Create comparison data
    comparison_data = []
    for word in top_words:
        comparison_data.append({
            'word': word,
            'count1': freq1.get(word, 0),
            'count2': freq2.get(word, 0)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_comparison['count1'],
        y=df_comparison['word'],
        orientation='h',
        name='Set 1',
        marker_color='blue'
    ))
    
    fig.add_trace(go.Bar(
        x=df_comparison['count2'],
        y=df_comparison['word'],
        orientation='h',
        name='Set 2',
        marker_color='red'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Frequency",
        yaxis_title="Word",
        barmode='group',
        template="plotly_white"
    )
    
    return fig