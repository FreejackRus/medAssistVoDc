import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface ClinicalRecommendation {
  id: string;
  title: string;
  description: string;
  annotation: string;
  publishdate: string;
  status: number;
  organization: string;
  mkb_code: string;
  category: string;
  age_group: string;
  keywords: string;
  version: string;
  code_version: string;
}

export interface RecsResponse {
  success: boolean;
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  recommendations: ClinicalRecommendation[];
}

export function useClinicalRecs(query: string, page: number, pageSize: number) {
  return useQuery<RecsResponse>({
    queryKey: ["clinical-recommendations", query, page, pageSize],
    queryFn: () => {
      const params = new URLSearchParams({
        q: query,
        page: String(page),
        page_size: String(pageSize),
      });
      return apiFetch<RecsResponse>(`/clinical-recommendations?${params}`);
    },
    placeholderData: keepPreviousData,
    staleTime: 5 * 60 * 1000,
  });
}
