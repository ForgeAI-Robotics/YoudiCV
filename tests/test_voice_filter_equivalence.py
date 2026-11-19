import math
import numpy as np
import pytest
from piper_handeye.voice_audio import lowpass_pcm16

# Reference recurrence preserves the previous arithmetic and output rounding.
def reference(pcm, cutoff):
    omega = 2*math.pi*cutoff/16000
    c = math.cos(omega); alpha = math.sin(omega)/math.sqrt(2)
    b0 = (1-c)/(2*(1+alpha)); b1 = (1-c)/(1+alpha)
    a1 = -2*c/(1+alpha); a2 = (1-alpha)/(1+alpha)
    x1=x2=y1=y2=0.
    result=[]
    for x in np.frombuffer(pcm,dtype='<i2'):
        x=int(x); y=b0*x+b1*x1+b0*x2-a1*y1-a2*y2
        result.append(max(-32768,min(32767,round(y))))
        x2,x1,y2,y1=x1,x,y1,y
    return np.array(result,dtype='<i2')

@pytest.mark.parametrize('cutoff',[100.,1000.,4000.,7000.])
def test_compiled_filter_matches_reference_to_one_pcm_unit(cutoff):
    rng=np.random.default_rng(25)
    for values in [rng.integers(-32768,32768,16000,dtype=np.int16),np.full(16000,32767,dtype=np.int16),np.zeros(16000,dtype=np.int16)]:
        actual=np.frombuffer(lowpass_pcm16(values.astype('<i2').tobytes(),cutoff),dtype='<i2')
        expected=reference(values.astype('<i2').tobytes(),cutoff)
        assert np.max(np.abs(actual.astype(int)-expected.astype(int))) <= 1

def test_empty_and_odd_length():
    assert lowpass_pcm16(b'')==b''
    with pytest.raises(ValueError): lowpass_pcm16(b'\x00')
