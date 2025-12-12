class FileManager:
    def __init__(self, nom_fichier='resultats.txt'):
        self.nom_fichier = nom_fichier
    
    def sauvegarder_resultats(self, resultats, classement):
        try:
            with open(self.nom_fichier, 'w', encoding='utf-8') as f:
                for i, (nom, res) in enumerate(classement, 1):
                    f.write(f"{i}. {nom}: {res['accuracy']:.4f}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("RÉSULTATS DÉTAILLÉS\n")
                f.write("="*60 + "\n\n")
                
                for nom, res in resultats.items():
                    f.write(f"{nom}:\n")
                    f.write(f"   Accuracy: {res.get('accuracy')}\n\n")
                    f.write(f"   Matrice de confusion:\n{res.get('confusion_matrix')}\n\n")
                    f.write(f"   Rapport:\n{res.get('rapport')}\n\n")

            print("Fichier créé avec succès :", self.nom_fichier)

        except Exception as e:
            print("Erreur pendant l’écriture :", e)
