import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from './types';

afterEach(() => vi.unstubAllGlobals());

describe('API client', () => {
  it('posts a SMILES string for RDKit inspection', async () => {
    const molecule = { smiles: 'CCO', valid: true, descriptors: { formula: 'C2H6O' } };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => molecule });
    vi.stubGlobal('fetch', fetchMock);
    await expect(api.inspect('CCO')).resolves.toMatchObject(molecule);
    expect(fetchMock).toHaveBeenCalledWith('/api/molecule', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ smiles: 'CCO' }),
    }));
  });

  it('surfaces chemistry validation errors from the API', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      json: async () => ({ detail: 'Invalid SMILES' }),
    }));
    await expect(api.inspect('broken')).rejects.toThrow('Invalid SMILES');
  });
});
