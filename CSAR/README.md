# \# CSAR-Net

# 

# \*\*Causal Survival Attention Residual Network\*\*

# 

# Novel architecture for student dropout survival analysis

# combining XGBoost base embedding, residual correction blocks,

# causal attention, and Cox survival output.

# 

# \## C-Index Results (n=36,924 — UCI + OULAD)

# 

# | Model | C-Index | 95% CI |

# |---|---|---|

# | \*\*CSAR-Net (Proposed)\*\* | \*\*0.8974\*\* | \[0.8909, 0.8970] |

# | XGBoost-Cox | 0.8924 | \[0.8898, 0.8956] |

# | Cox PH | 0.8871 | \[0.8845, 0.8897] |

# | DeepHit | 0.8816 | — |

# | RSF | 0.8764 | — |

# | DSM | 0.7575 | — |

# 

# \## Ablation Study

# 

# | Variant | C-Index | Drop |

# |---|---|---|

# | Full CSAR-Net | 0.8992 | — |

# | No XGB Embedding | 0.8961 | −0.003 |

# | No Causal Attention | 0.8960 | −0.003 |

# | No Residual Blocks | 0.8735 | −0.026 |

# 

# \## Datasets

# \- UCI Student Dropout (Portugal, n=4,424)

# \- OULAD Open University (UK, n=32,500)

# \- College Math MEC Oman (n=1,896)

# \- MEC Oman M7Final (n=1,655)

# 

# \## Citation

# Paper under review — Applied Intelligence (Springer).

