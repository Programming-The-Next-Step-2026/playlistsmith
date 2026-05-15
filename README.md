# Playlistsmith
This project is about creating a package (and associated GUI) that allows users to provide a list of songs which is then split into playlists based on song features. 

## Brief Summary
Overall, the package is supposed to provide a user-friendly, automated way to create playlists of similar songs given an unsorted track list. It is motivated by my own habit of just putting every song I liked once into a single playlist. Over the years, this playlist has become huge and manually splitting it into sub-playlists is just unfeasible.

Hopefully, this package will make this task easy and accessible not only to me but most of all other users.

## Step-By-Step Overview
The project can be devided into several overarching steps:

1. Standardized input format 
    - The package requires some standardized input for the track lists/playlists. (artist, track) pairs seem like a reasonable choice.
    - To make the package more user-friendly, I am planning to find a way to extract these pairs from, for example, Spotify or other platforms. However, this entails implementing user authentification and some legal considerations that I need to take into account ([see here](https://developer.spotify.com/terms#section-iv-restrictions)).

2. Obtain features from songs
    - Next, I need to implement some way to obtain song features. The features form the basis for the classification into playlists.
    - Here, it would be easiest to use an existing API (e.g., [ReccoBeats](https://reccobeats.com/)) that allows querries based on song names and artists and then returns features.
    - Alternatively, one could access freely available 30 sec snippets of songs and then use an API for feature extraction based on these audio snippets. This, however, seems to be an extension that I can consider if there is extra time.

3. Create playlists based on song features
    - I am planning to use statistical techniques to assign tracks to playlists based on their features. I will start with clustering techniques (e.g., k-means).
    - Given some extra time and access to song snippets, it would also be interesting to use a pretrained neural network that takes audio snippets as input for this classification task.

4. Writing playlists into account
    - Given access to a streaming account, there should be a feature to add the created playlists to the user's account.

5. User-friendly GUI
    - To make usage user-friendly and simplify presentation, all this should be implemented with a GUI.
    - Examples for GUI elements include: 
        - Upload track list
        - Feature extraction (with progress bar for visual feedback)
            - Optional: Download track snippets 
        - Selection of clustering/machine learning algorithm to classify songs
        - Hyperparameter tuning (e.g., k in k-means)
        - Output visualisation (e.g., interactive inspection of clusters)
        - Option to add playlists to account

## Disclaimer
This project is not affiliated with, sponsored by, or endorsed by Exportify or Spotify. These names are used only to describe compatible input formats and data sources.

