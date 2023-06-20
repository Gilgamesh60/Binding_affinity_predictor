from src.functions import *
import argparse 
import logging

model = torch.load("/content/Binding_affinity_predictor/best_model.pt")
                 
  
 if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--ligand_file", help="Enter your ligand file in sdf format here",default='')
    parser.add_argument("-p", "--protein_file", help="Enter your protein file in pdb format here",default='')
    args= parser.parse_args()
    complex_graph = interaction_graph(args.ligand_file,args.protein_file)
    affinity_preds = model(complex_graph)
    logger.info("Our model calculates the binding affinity as −log(ki/kd) where ki is inhibition constant and kd is the equilibrium constant")
    logger.info(f"So by the above logic, the binding affinity of your protein-ligand complex is {affinity_pred}")
    if affinity_preds>=10:
      logger.info(" The binding affinity of your complex is extremely high. These two are very competant with each other")
    elif 7<=affinity_preds<10:
      logger.info("The binding affinity of your complex is high.")
    elif 5<=affinity_preds<7:
      logger.info('The binding affinity of your complex is moderate.Not too high not too low")
    else:
       logger.info('The binding affinity of you complex is pretty low.")
     
      
    
    
        
