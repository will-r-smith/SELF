import torch
from config.load_config import load_config
from copy import deepcopy
from src.matrix_utils import relative_error, sorted_mat, prune, do_lr, do_mm


class AbstractIntervention:

    # Helper functions for matrix update
    @staticmethod
    def get_parameter(model, name):
        for n, p in model.named_parameters():
            if n == name:
                return p
        raise LookupError(name)

    @staticmethod
    def update_model(model, name, params):
        with torch.no_grad():
            AbstractIntervention.get_parameter(model, name)[...] = params

    @staticmethod
    def get_edited_model(model, args, in_place=True):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        if args.lname == "dont":
            print(f"No intervention.")
            return model_edit

        # Load configuration
        config = load_config()
        if args.lname != "None":
            try: 
                converted_name = config[args.model]["naming_conv"]["layers"][args.lname]
            except:
                raise AssertionError(f"Unhandled name: '{args.lname}'")
        else:
            converted_name = "none"

        num_update = 0
        for name, param in model.named_parameters():

            param.requires_grad = False

            if not name.startswith(f"{config[args.model]["naming_conv"]["base"]}{args.lnum}"):
                # if it isn't the selected layer number then skip
                continue

            if args.lname != "none" and not name.startswith(f"{config[args.model]["naming_conv"]["base"]}{args.lnum}.{converted_name}"):
                # if a specified layer is named (i.e. "none") then perform only for that named layer name and number
                # so skip if it isn't the right layer name 
                continue
                # otherwise the method will proceed for all layer types at the specified layer number

            # For the sparsity analysis
            mat_analysis = param.detach().cpu().numpy()
            mat_sort = sorted_mat(mat_analysis)

            mat_analysis = param.detach().cpu().numpy()
            mat_analysis_tensor = deepcopy(param)

            if args.intervention == 'dropout':
                mat_analysis = prune(mat_analysis, mat_sort, args.rate)  # pruned_mat
                mat_analysis = torch.from_numpy(mat_analysis)

            elif args.intervention == 'lr':
                model_edit, mat_analysis = do_lr(model, name, mat_analysis_tensor.type(torch.float32), (1 - args.rate))

            elif args.intervention == 'zero':
                mat_analysis_tensor = deepcopy(param)
                mat_analysis = 0.0 * mat_analysis_tensor.type(torch.float32)

            elif args.intervention == 'mm':
                mat_analysis = do_mm(mat_analysis_tensor.type(torch.float32))

            else:
                raise AssertionError(f"Unhandled intervention type {args.intervention}")

            frobenius_ratio = relative_error(mat_analysis_tensor.type(torch.float32), mat_analysis)

            #AbstractIntervention.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update == 1, f"Was supposed to make 1 update to the model but instead made {num_update} updates."
        
        return model_edit, frobenius_ratio