import matplotlib.pyplot as plt
import os

class DataVisualizer:
    
    def __init__(self, data):
        self.data = data        
        if not os.path.exists('figures'):
            os.makedirs('figures')
    
    def distribution_age(self):
        plt.figure(figsize=(10, 6))        
        plt.hist(self.data["Age"], color = "blue")
        plt.title("Distribution d'age")
        plt.xlabel("Age")
        plt.ylabel("Frequence")
        plt.grid(True, alpha=0.2)
        plt.savefig("figures/distribution_age.png")
        plt.show()
    
    def distribution_imc(self):
        plt.figure(figsize=(10, 6))        
        plt.hist(self.data["BMI"], color = "red")
        plt.title("Distribution de bmi")
        plt.xlabel("IMC")
        plt.ylabel("Frequence")
        plt.grid(True, alpha=0.2)
        plt.savefig("figures/distribution_imc.png")
        plt.show()
    
    def relation_age_imc(self):
        plt.figure(figsize=(10, 6))        
        plt.scatter(self.data["Age"], self.data["BMI"], color = "green", alpha=0.3)
        plt.title("Relation entre age et bmi")
        plt.xlabel("Age")
        plt.ylabel("IMC")
        plt.grid(True, alpha=0.2)
        plt.savefig("figures/relation_age_imc.png")
        plt.show()
    
    def repartition_diabete(self):        
        plt.figure(figsize=(10, 6))
        comptage = self.data["Diabetes_Status"].value_counts()
        plt.bar(['Non diabétique', 'Diabétique'], comptage.values, color = "purple")
        plt.title("Repartition du diabete")
        plt.xlabel("Diabete")
        plt.ylabel("Frequence")
        plt.grid(True, alpha=0.2)
        plt.savefig("figures/repartition_diabete.png")
        plt.show()
