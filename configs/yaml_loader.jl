using YAML

function load_yaml_iter(PATH =  string(@__DIR__) * "/config.yaml")
    config_dict = YAML.load_file(PATH)
    config = (; (Symbol(k) => v for (k,v) in config_dict)...)
    if "defaults" in collect(keys(config_dict))
        @info "Loading nested YAML"
        for i in 1:size(config_dict["defaults"],1)
            key = collect(keys(config_dict["defaults"][i]))[1]
            value = collect(values(config_dict["defaults"][i]))[1]
            NEW_PATH = string(@__DIR__) * "/$(key)/$(value).yaml" 
            config_dict[key], config_tmp = load_yaml_iter(NEW_PATH)
            config = merge(config,(; (Symbol(key) => config_tmp)))
            @info i, key
        end
    end
    config, config_dict
end
