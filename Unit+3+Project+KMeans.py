
# coding: utf-8

# In[1]:


import sys
print(sys.version)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[3]:


# Generating Data - 2 Means
mean1 = [np.random.randint(100), np.random.randint(100)]
mean2 = [np.random.randint(100), np.random.randint(100)]

# Make Diagonal CoVariance 

cov = [[100,0],[0,100]]

x1, y1 = np.random.multivariate_normal(mean1, cov, 100).T
x2, y2 = np.random.multivariate_normal(mean2, cov, 100).T


x = np.append(x1,x2)
y = np.append(y1,y2)

plt.plot(x,y, 'x')
plt.axis('equal')
plt.show()


# In[4]:


# Generating Data - 2 Means
mean1 = [np.random.randint(50), np.random.randint(50)]
mean2 = [np.random.randint(50), np.random.randint(50)]

# Make Diagonal CoVariance 

cov = [[100,0],[0,100]]

x1, y1 = np.random.multivariate_normal(mean1, cov, 100).T
x2, y2 = np.random.multivariate_normal(mean2, cov, 100).T


x = np.append(x1,x2)
y = np.append(y1,y2)

plt.plot(x,y, 'x')
plt.axis('equal')
plt.show()


# In[5]:


X = np.array(zip(x,y))


# In[19]:


#Actual Making KMeans Model 

kmeans = KMeans(n_clusters = 2)

#Try n_cluster with  different number




# In[21]:


X = np.array(list(zip(x,y)))


# In[22]:


#Actual Making KMeans Model 

kmeans = KMeans(n_clusters = 2)

#Try n_cluster with  different number




# In[23]:


kmeans.fit(X)


# In[24]:


centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(labels)
print(centroids)


# In[25]:


colors = ["g.", "y."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 15)
    
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'X', s = 150, zorder = 10)

plt.show()


# In[26]:


colors = ["r.", "y."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 15)
    
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'X', s = 150, zorder = 10)

plt.show()


# In[27]:


colors = ["r.", "y."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 5)
    
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'X', s = 150, zorder = 10)

plt.show()


# In[28]:


print(centroids)
print(mean1, mean2)


# In[29]:


#Actual Making KMeans Model 
#Trying with different clusters ...
kmeans = KMeans(n_clusters = 6)




# In[30]:


kmeans.fit(X)


# In[31]:


centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(labels)
print(centroids)


# In[32]:


colors = ["g.", "y.", "r.", "b.", "c.", "b."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 15)
    
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'X', s = 150, zorder = 10)

plt.show()


# In[33]:


colors = ["g.", "y.", "r.", "b.", "c.", "b."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 5)
    
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'X', s = 150, zorder = 10)

plt.show()


# In[34]:


#pretty Cool


# In[35]:


print(centroids)
print(mean1, mean2)

