import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal

def k_means(n, dic, minx, maxx, miny, maxy):
	cluster = {}
	for i in range(n):
		cluster.setdefault(i, {'coord':[random.uniform(minx, maxx), random.uniform(miny, maxy)], 'dic_lst' : []})
	change = True
	while change:
		for i in range(n):
			cluster[i]['dic_lst'] = []
		change = False
		for i in dic:
			prev_clus = dic[i]['cluster']
			dist_lst = [0] * n
			for j in cluster:
				dist_lst[j] = calc_distance(dic[i]['coord'], cluster[j]['coord'])
			min_dist = min(dist_lst)
			clus = dist_lst.index(min_dist)
			dic[i]['cluster'] = clus
			cluster[clus]['dic_lst'].append(i)
			if clus != prev_clus:
				change = True
		for j in range(n):
			cluster[j]['coord'] = calc_centroid(dic, cluster[j]['dic_lst'], cluster[j]['coord'])
	plot_graph(dic, cluster,n)

def plot_graph(dic, cluster, n):
	color_lst = ['y', 'r', 'g', 'c', 'b']
	for j in cluster:
		lst = cluster[j]['dic_lst']
		for p in lst:
			plt.scatter(float(dic[p]['coord'][0]), float(dic[p]['coord'][1]), c = color_lst[j])
	title = "For k = "+str(n)
	plt.title(title)
	plt.show()

def calc_centroid(dic, lst, orig):
	n = len(lst)
	if n > 0:
		x = 0.0
		y= 0.0
		for p in lst:
			x += float(dic[p]['coord'][0])
			y += float(dic[p]['coord'][1])
		return [x/n, y/n]
	else: 
		return orig


def calc_distance(lst1, lst2):
	x = float(lst1[0]) - lst2[0]
	y = float(lst1[1]) - lst2[1]
	return math.sqrt(x**2 + y**2)

def em(df):
	n = 3
	pi = [1.0/3,1.0/3,1.0/3]
	cov_dic = {}
	for i in range(3):
		cov_dic.setdefault(i, np.identity(2))
	choice = np.random.choice(len(df.index), 3, replace = False)
	mean = []
	for ch in choice:
		mean.append([df.loc[ch]['x'],df.loc[ch]['y']]) 
	##Calculate r
	change = True
	prev_like = 0
	log_lst = []
	while change:
		total_sum = [0] * len(df.index)
		r = pd.DataFrame(columns = ['0','1','2'])
		for i in range(len(df.index)):
			row = []
			x = [df.loc[i]['x'],df.loc[i]['y']]
			sum_row = 0
			for j in range(3):
				sum_row += pi[j] * multivariate_normal.pdf(x, mean=mean[j], cov=cov_dic[j])
			total_sum[i] = sum_row
			for j in range(3):
				row.append(pi[j] * multivariate_normal.pdf(x, mean=mean[j], cov=cov_dic[j]) / sum_row)
			r.loc[i] = row
		####Calculate mc#######
		mc = [0] * 3
		sum_clus = [0] * 3
		for i in range(n):
			mc[i] =  r.loc[:, str(i)].sum()	
			pi[i] = mc[i]/len(df.index)
			###update mean[i]
			sum_clus[i] = [np.dot(r[str(i)],df['x']),(np.dot(r[str(i)] ,df['y']))]
			mean[i] = np.divide(sum_clus[i],mc[i])
			##update covariance
			x = (((df['x'] - mean[i][0])**2) * r[str(i)]).sum()  
			xy = (((df['x'] - mean[i][0]) * (df['y'] - mean[i][1])) * r[str(i)]).sum()
			yx = xy
			y = (((df['y'] - mean[i][1])**2) * r[str(i)]).sum()
			cov_dic[i] = [np.divide([x,xy], mc[i]),np.divide([yx,y], mc[i])]
		likelihood = np.log(total_sum).sum()
		print likelihood
		if prev_like == likelihood:
			change = False
		prev_like = likelihood
		log_lst.append(likelihood)
	plt.plot(range(len(log_lst)),log_lst)
	return [likelihood, r, mean, cov_dic,pi]

def kernel_k_means(kernel, df):
	choice = np.random.choice(len(df.index), 2, replace = False)
	cluster = []
	for ch in choice:
		cluster.append([df.loc[ch]['x'],df.loc[ch]['y']])
	zero_lst = []
	one_lst = []
	clus_assgn = []
	for i in range(len(df.index)):
		zero_lst.append(calc_distance(df.loc[i], cluster[0]))
		one_lst.append(calc_distance(df.loc[i], cluster[1]))
		if zero_lst[i] > one_lst[i]:
			clus_assgn.append('1')
		else:
			clus_assgn.append('0')
	df['0'] = zero_lst
	df['1'] = one_lst
	df['Cluster'] = clus_assgn
	change = True
	while change:
		for j in [0,1]:
			clus_list = df.loc[df["Cluster"] == str(j)].index.tolist()
			sum_t = 0
			for x in clus_list:
				for y in clus_list:
					sum_t += kernel.loc[x][y]
			dist = []
			for i in range(len(df.index)): 
				f = kernel.iloc[i][str(i)]
				s = (2 * kernel.iloc[clus_list][str(i)].sum()) / len(clus_list)
				t = sum_t / (len(clus_list)**2)
				dist.append(f - s + t)
			df[str(j)] = dist
		new = df[['0','1']].idxmin(axis=1)
		if df["Cluster"].equals(new):
			change = False
		else:
			df["Cluster"] = new
	color_lst = ['r', 'g']
	for i in range(len(df.index)):
		color = int(df.loc[i]['Cluster'])
		x = df.loc[i]['x']
		y = df.loc[i]['y']
		plt.scatter(x, y, c = color_lst[color]) 
	plt.show()
	

if __name__ =='__main__':
	f = open('hw5_circle.csv', 'r')
	reader = csv.reader(f, delimiter=',')
	dic = {}
	i = 0
	minx = float("inf")
	maxx = float("-inf")
	miny = float("inf")
	maxy = float("-inf")
	for row in reader:
		minx = min(minx, float(row[0]))
		maxx = max(maxx, float(row[0]))
		miny = min(miny, float(row[1]))
		maxy = max(maxy, float(row[1]))
		dic.setdefault(i, {'coord': row, 'cluster': -1})
		i += 1    
	for i in [2,3,5]:
		k_means(i, dic, minx, maxx, miny, maxy)
	f.close()


	f = open('hw5_blob.csv', 'r')
	reader = csv.reader(f, delimiter=',')
	dic = {}
	i = 0
	minx = float("inf")
	maxx = float("-inf")
	miny = float("inf")
	maxy = float("-inf")
	for row in reader:
		minx = min(minx, float(row[0]))
		maxx = max(maxx, float(row[0]))
		miny = min(miny, float(row[1]))
		maxy = max(maxy, float(row[1]))
		dic.setdefault(i, {'coord': row, 'cluster': -1})
		i += 1    
	for i in [2,3,5]:
		k_means(i, dic, minx, maxx, miny, maxy)
	f.close()
	# #######Kernel K means
	df = pd.read_csv("hw5_circle.csv",header=None)
	df.columns = ['x','y']
	kernel = pd.DataFrame()
	sigma = 1
	for i in range(len(df.index)):	
		f = (df * df.loc[i]).sum(axis = 1)
		s = (df**2 * 7).sum(axis = 1) * (df.loc[i]**2 * 7).sum()
		kernel[str(i)] = f + s
	kernel_k_means(kernel, df)
	#####EM algorithm############
	df = pd.read_csv("hw5_blob.csv",header=None)
	df.columns = ['x','y']
	max_l = float("-inf")
	for i in range(5):
		lst = em(df)
		print "For run ",i
		print "Likelihood", lst[0]
		print "Cluster Coordinates ",lst[2]
		print "Covariance Matrix" , lst[3]
		print "Prior Probabilities", lst[4],"\n"
		if lst[0] > max_l:
			max_l = lst[0]
			r = lst[1]
			mean = lst[2]
			cov = lst[3]
	print "Best Likelihood ",max_l 
	plt.show()
	color_lst = ['y', 'r', 'g']
	for i in range(len(df.index)):
		r_lst = r.loc[i].tolist()
		color = r_lst.index(max(r_lst))
		x = df.loc[i]['x']
		y = df.loc[i]['y']
		plt.scatter(x, y, c = color_lst[color]) 
	plt.show()