feature_importances = [('(D D.T D)_st', 0.2249942841778843),
 ('(DD D.T)_st', 0.15264564748068635),
 ('tf-idf', 0.09060615522092176),
 ('(D.T DD)_st', 0.08910742248029349),
 ('(DD)_st', 0.06415691281351038),
 ('(DDD)_st', 0.05473062494534976),
 ('year', 0.04521757548233593),
 ('deg_in(t)', 0.03936042656993067),
 ('(D D.T)_st', 0.03508444562991492),
 ('(D.T D)_st', 0.034197455014868004),
 ('authors_connect', 0.02724425444233442),
 ('f_AA', 0.02485148509510255),
 ('title', 0.016844238267126756),
 ('deg_out(s)', 0.01185590868864163),
 ('deg_out(t)', 0.011078793623308708),
 ('Doc2Vec', 0.009859200316258265),
 ('deg_in(s)', 0.007567525831598499),
 ('f_AtA', 0.007476797498544422),
 ('f_AAt', 0.007018199517756619),
 ('a_deg_in_tgt', 0.005931465167679303),
 ('f_AAAt', 0.005771751471457777),
 ('a_deg_out_src', 0.005693521842693969),
 ('f_AAtA', 0.00551831518939777),
 ('a_deg_in_src', 0.00548548304950041),
 ('f_AAA', 0.00496114867871424),
 ('f_AtAA', 0.004839722876504906),
 ('a_deg_out_tgt', 0.004269116653834148),
 ('authors', 0.002408455409842574),
 ('journals', 0.001133089173765737),
 ('title_cited', 9.057739024180153e-05)]


# Plot variable importance
import matplotlib.pyplot as plt

# list of x locations for plotting
x_values = list(range(len(feature_importances)))

# Make a bar chart
plt.bar(x_values, [importance for (_, importance) in feature_importances], orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, [name for (name, _) in feature_importances], rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances'); plt.show()