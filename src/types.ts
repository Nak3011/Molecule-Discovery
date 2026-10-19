export type Descriptors = { formula: string; molecularWeight: number; exactMass: number; logp: number; hbd: number; hba: number; tpsa: number; rotatableBonds: number; rings: number; heavyAtoms: number; qed: number };
export type Geometry = { molblock: string; atoms: number; bonds: number; heavyAtoms: number; energy: number | null; forceField: string; coordinateSpan: {x:number;y:number;z:number}; has3d: boolean };
export type Molecule = { id?: string; name?: string | null; smiles: string; inputSmiles?: string; originalOutput?: string; relatedReference?: boolean; valid: boolean; descriptors: Descriptors | null; geometry?: Geometry; source?: string };
export type Batch = { candidates: Molecule[]; metrics: {total:number; valid:number; validity:number; unique:number; uniqueness:number; averageMolecularWeight:number; averageLogP:number}; modelMetrics?: {total:number; valid:number; validity:number; unique:number; uniqueness:number; averageMolecularWeight:number; averageLogP:number}; source: string; relatedCount?:number };
export type LatentPoint = {name:string; smiles:string; x:number; y:number};

const API = import.meta.env.VITE_API_URL || '';
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'The request could not be completed.');
  return data as T;
}
export const api = {
  status: () => request<{mode:string;model:{loaded:boolean;reason:string;latentDimension:number|null}}>('/api/status'),
  inspect: (smiles:string) => request<Molecule>('/api/molecule', {method:'POST',body:JSON.stringify({smiles})}),
  generate: (count:number, temperature:number) => request<Batch>('/api/generate', {method:'POST',body:JSON.stringify({count,temperature})}),
  geometry: (smiles:string) => request<Geometry>('/api/geometry', {method:'POST',body:JSON.stringify({smiles})}),
  latent: () => request<{points:LatentPoint[]; illustrative:boolean; source?:string; projection?:string}>('/api/latent'),
  interpolate: (a:string,b:string,steps:number) => request<{steps:Array<{alpha:number;smiles:string;valid:boolean}>;source:string}>('/api/interpolate', {method:'POST',body:JSON.stringify({smilesA:a,smilesB:b,steps})}),
  exportUrl: (format:'csv'|'sdf', smiles:string[], sources:string[] = []) => {
    const query = new URLSearchParams({smiles:smiles.join('\n')});
    if (sources.length) query.set('sources', sources.join(','));
    return `${API}/api/export/${format}?${query.toString()}`;
  },
};
