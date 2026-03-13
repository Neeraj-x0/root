def MakeKerasConvTranspose(layer):
    """
    Create a Keras-compatible Conv2DTranspose layer operation using SOFIE framework.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  data type (must be float), weight and bias name, kernel size,
                  dilations, padding, strides and output_padding.

    Returns:
    ROperator_ConvTranspose: A SOFIE framework operator representing the Conv2DTranspose operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    attributes = layer["layerAttributes"]
    fWeightNames = layer["layerWeight"]
    fKernelName = fWeightNames[0]
    fBiasName = fWeightNames[1] if len(fWeightNames) > 1 else ""

    fAttrDilations    = list(attributes["dilation_rate"])
    fAttrGroup        = 1  # Conv2DTranspose in Keras does not support groups
    fAttrKernelShape  = list(attributes["kernel_size"])
    fAttrStrides      = list(attributes["strides"])
    fKerasPadding     = str(attributes["padding"])
    raw_output_padding = attributes.get("output_padding") or [0, 0]
    fAttrOutputPadding = list(raw_output_padding)
    fAttrOutputShape  = []
    fAttrPads         = []

    if fKerasPadding == "valid":
        fAttrAutopad = "VALID"
    elif fKerasPadding == "same":
        fAttrAutopad = "SAME_UPPER"
    else:
        raise RuntimeError(
            "TMVA::SOFIE - RModel Keras Parser doesn't yet support Conv2DTranspose with padding " + fKerasPadding
        )

    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_ConvTranspose["float"](
            fAttrAutopad,
            fAttrDilations,
            fAttrGroup,
            fAttrKernelShape,
            fAttrOutputPadding,
            fAttrOutputShape,
            fAttrPads,
            fAttrStrides,
            fLayerInputName,
            fKernelName,
            fBiasName,
            fLayerOutputName,
        )
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Conv2DTranspose does not yet support input type " + fLayerDType
        )
