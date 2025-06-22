# Evaluation on 3 medical models

- **Medmamba**
- **MedViT**
- **MedViTV2**

## How to use?

1. clone [MedMamba](#https://github.com/YubiaoYue/MedMamba.git), [MedViT](#https://github.com/Omid-Nejati/MedViT.git) and [MedViTV2](#https://github.com/Omid-Nejati/MedViTV2.git) into this project

   - Structure

   ```
    - medmodel_eval
    --- MedMamba
    --- MedViT
    --- MedViTV2
    --- model_eval
   ```

2. Corresponding **conda** environments

   - **MedMamba**: `medmamba`
   - **MedViT**: `medvitv2-ly`
   - **MedViTV2**: `medvitv2-ly`

3. Starting training:
   1. Open a run\_{modelname}.sh
   2. Fill in wandb API key and entity
   3. Fill in DATASETS
