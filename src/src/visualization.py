def Visualize2dSegmentation(classifier):
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    color_map = {}
    
    X = classifier.X
    labels = classifier.labels

    #boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    
    #create a color mapping for labels
    count = -1

    for label in labels:
        if str(label) in color_map:
            m_val = color_map.get(str(label))
        else: 
            count += 1
            m_val = count
        color_map[str(label)] = m_val
        
    mapped_labels = []
    for label in  labels:
        mapped_labels = mapped_labels + [color_map[str(label)]]        
      
    #Draw decision boundaries
    mapped_predictions = []


    xx, yy = np.meshgrid(np.arange(x_min, x_max),
                         np.arange(y_min, y_max))
    
    predictions = classifier.Predict(np.c_[xx.ravel(), yy.ravel()])
    predictions = predictions.reshape(-1, 1)
    
    
    for label in np.nditer(predictions[:,0]):
        mapped_predictions = mapped_predictions + [color_map[str(label)]]
        
   
    
    mapped_predictions = np.array(mapped_predictions)
    mapped_predictions = mapped_predictions.reshape(xx.shape)

    plt.pcolormesh(xx, yy, mapped_predictions, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c = mapped_labels, cmap=cmap_light)

    plt.xlim(x_min , x_max)
    plt.ylim(y_min, y_max)
    plt.legend(numpoints=1)

    plt.title("Scatter Classification")
    
    plt.show()