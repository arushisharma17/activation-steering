

```bash
python --version   #to check python version
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/IBM/activation-steering
cd activation-steering
pip install -e activation-steering
```

### Running initial experiments-

#### Getting the tssb-3m dataset 
[Dataset](https://github.com/cedricrupb/TSSB3M?tab=readme-ov-file#datasets)

I have a filtered version of the dataset with just the following bugs:

- `MORE_SPECIFIC_IF`  
- `ADD_METHOD_CALL`  
- `ADD_FUNCTION_AROUND_EXPRESSION`  
- `SAME_FUNCTION_MORE_ARGS`  
- `SAME_FUNCTION_LESS_ARGS`  
- `CHANGE_BINARY_OPERATOR`  
- `CHANGE_COMPARISON_OPERATOR`  
- `SINGLE_TOKEN`

The filtered dataset is located at docs/demo-data as filtered-0, filtered-1, filtered-2. filtered-0 was used to create the apr questions and contrastive pairs (mutually exclusive sets) using apr.sh. No need to re-run. 

#### To run initial experiment
```bash
sbatch run_apr_eval.sh
```




